"""Offline USD inspection: mesh sections, compound collider and grasp candidates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import sysconfig

import numpy as np
from geometry import PROFILE, candidate, ray_radius, sectors, preflight

REPO = Path(__file__).resolve().parents[2]


def usd_imports():
    # Use the USD distribution bundled with the selected Isaac environment, without Kit.
    site = Path(sysconfig.get_paths()['purelib'])
    libs = sorted((site/'isaacsim/extscache').glob('omni.usd.libs-*/bin'))
    if libs and str(libs[0]) not in os.environ.get('LD_LIBRARY_PATH','').split(':'):
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = ':'.join([str(Path(sys.prefix)/'lib'),str(libs[0]),env.get('LD_LIBRARY_PATH','')])
        env['PYTHONPATH'] = str(libs[0].parent)+os.pathsep+env.get('PYTHONPATH','')
        os.execve(sys.executable,[sys.executable,*sys.argv],env)
    if libs:
        sys.path.insert(0,str(libs[0].parent))
    from pxr import Usd, UsdGeom, UsdPhysics
    return Usd, UsdGeom, UsdPhysics


def mesh_from_usd(path, scale, Usd, UsdGeom):
    import trimesh
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    meshes, paths = [], []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.array([cache.GetLocalToWorldTransform(prim).Transform(p)
                           for p in mesh.GetPointsAttr().Get()])*scale*UsdGeom.GetStageMetersPerUnit(stage)
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        faces, offset = [], 0
        for count in counts:
            face = indices[offset:offset+count]
            faces.extend([[face[0],face[i],face[i+1]] for i in range(1,count-1)])
            offset += count
        meshes.append(trimesh.Trimesh(points,np.asarray(faces),process=False))
        paths.append(str(prim.GetPath()))
    if not meshes:
        raise ValueError('No mesh found in peg USD')
    return trimesh.util.concatenate(meshes), paths


def area(points):
    xy = np.asarray(points)[:,:2]
    return abs(np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1)))/2


def contours(mesh,z):
    section = mesh.section(plane_origin=[0,0,z],plane_normal=[0,0,1])
    paths = sorted(section.discrete,key=area,reverse=True) if section is not None else []
    if len(paths) != 2:
        raise ValueError(f'Expected an outer contour and one hole at z={z}; found {len(paths)}')
    if any(np.linalg.norm(p[0]-p[-1]) > 1e-7 for p in paths):
        raise ValueError('Open mesh section')
    return [p[:-1,:2] for p in paths]


def robot_audit(path,Usd,UsdGeom,UsdPhysics):
    from scipy.spatial.transform import Rotation
    stage = Usd.Stage.Open(str(path)); cache = UsdGeom.XformCache()
    tip = stage.GetPrimAtPath('/psm/psm_tool_tip_link')
    inverse = cache.GetLocalToWorldTransform(tip).GetInverse()
    pads = []
    for number in (1,2):
        name = f'psm_tool_gripper{number}'
        prim = stage.GetPrimAtPath(f'/psm/{name}_link/collisions_xform/collisions')
        transform = cache.GetLocalToWorldTransform(prim)
        points = np.array([inverse.Transform(transform.Transform(p)) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()])
        joint = stage.GetPrimAtPath(f'/psm/psm_tool_yaw_link/{name}_joint')
        body = stage.GetPrimAtPath(UsdPhysics.Joint(joint).GetBody0Rel().GetTargets()[0])
        body_transform = cache.GetLocalToWorldTransform(body)
        pivot = np.array(inverse.Transform(body_transform.Transform(joint.GetAttribute('physics:localPos0').Get())))
        q = joint.GetAttribute('physics:localRot0').Get()
        v = np.array(q.GetImaginary()); w = q.GetReal()
        x = np.array([1.,0.,0.]); axis = x+2*np.cross(v,w*x+np.cross(v,x))
        from pxr import Gf
        axis = np.array(inverse.TransformDir(body_transform.TransformDir(Gf.Vec3d(*axis))))
        axis /= np.linalg.norm(axis)
        limits = [joint.GetAttribute(f'physics:{side}Limit').Get() for side in ('lower','upper')]
        pads.append(dict(points=points,pivot=pivot,axis=axis,limits_deg=limits))
    gaps = {}
    for angle in (0.,.07,.18,.45,.5):
        points = [Rotation.from_rotvec(pad['axis']*angle*sign).apply(pad['points']-pad['pivot'])+pad['pivot']
                  for pad,sign in zip(pads,(-1,1))]
        gaps[str(angle)] = float((points[1][:,0].min()-points[0][:,0].max())*1000)
    return dict(asset_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                jaw_geometry_tip_frame=[dict(points_m=p['points'].tolist(),pivot_m=p['pivot'].tolist(),
                                             joint_axis=p['axis'].tolist()) for p in pads],
                upright_pad_bounds_tip_frame_m=[np.array([p['points'].min(0),p['points'].max(0)]).tolist() for p in pads],
                jaw_limits_deg=[p['limits_deg'] for p in pads],
                minimum_x_gap_mm_by_angle_rad=gaps,
                note='Bounding gap over entire pads at rest orientation; not a contact-force model')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--peg_usd',type=Path,default=REPO/'data/Isaac/Healthcare/0.5.0/132c82d/Props/PegBlock/block.usd')
    p.add_argument('--robot_usd',type=Path,default=Path.home()/'.cache/ov/client/https/d36f3a4d2205f4288e3dfc1961a891b4254884432f41d7822f54437aca9b2d45.usd')
    p.add_argument('--rays',type=int,default=96)
    p.add_argument('--output',type=Path,default=PROFILE)
    args = p.parse_args()
    if args.rays < 48 or args.rays % 3:
        p.error('--rays must be >=48 and divisible by 3')
    Usd,UsdGeom,UsdPhysics = usd_imports()
    mesh,paths = mesh_from_usd(args.peg_usd,.011,Usd,UsdGeom)
    if not mesh.is_watertight:
        raise ValueError('Mesh is not watertight; repair before creating grasp colliders')
    lo,hi = mesh.bounds[:,2]
    center = np.array([.00578638,0.])
    angles = np.linspace(0,2*np.pi,args.rays,endpoint=False)
    # Resolve the ~0.5 mm rounded lips instead of extruding one section through them.
    offsets = [.000005,.00005,.0002,.00055]
    heights = sorted(set([lo+x for x in offsets]+[hi-x for x in offsets]))
    levels = []
    for z in heights:
        outer,inner = contours(mesh,z)
        levels.append(dict(z_m=float(z),inner_radius_m=[ray_radius(inner,center,a) for a in angles],
                           outer_radius_m=[ray_radius(outer,center,a) for a in angles]))
    levels[0]['z_m'],levels[-1]['z_m'] = float(lo),float(hi)
    profile = dict(schema='peg-contact-geometry-2',asset_sha256=hashlib.sha256(args.peg_usd.read_bytes()).hexdigest(),
                   scale=.011,mesh_paths=paths,vertices=len(mesh.vertices),triangles=len(mesh.faces),
                   bounds_m=mesh.bounds.tolist(),hole_center_m=center.tolist(),angles_rad=angles.tolist(),levels=levels,
                   approximation='Layered convex radial cells; rounded lips sampled, not an exact triangle mesh')
    from scipy.spatial import ConvexHull
    # Measure actual cooked-cell hull boundary against mesh at midpoints in angle AND height.
    errors = []
    cells = sectors(profile)
    for k,(low,high) in enumerate(zip(levels,levels[1:])):
        z = (low['z_m']+high['z_m'])/2
        outer,inner = contours(mesh,z)
        for i,a in enumerate(angles):
            direction = np.array([np.cos(a+np.pi/args.rays),np.sin(a+np.pi/args.rays),0])
            origin = np.array([*center,z])
            equations = ConvexHull(cells[k*args.rays+i]).equations
            num = -(equations[:,:3]@origin+equations[:,3]); den = equations[:,:3]@direction
            enter = max(num[den < -1e-12]/den[den < -1e-12])
            leave = min(num[den > 1e-12]/den[den > 1e-12])
            errors += [abs(enter-ray_radius(inner,center,a+np.pi/args.rays)),
                       abs(leave-ray_radius(outer,center,a+np.pi/args.rays))]
    profile['sampled_max_radial_error_mm'] = float(max(errors)*1000)
    if profile['sampled_max_radial_error_mm'] > .20:
        raise ValueError('Collider approximation error exceeds 0.20 mm; increase resolution')
    profile['robot'] = robot_audit(args.robot_usd,Usd,UsdGeom,UsdPhysics) if args.robot_usd.is_file() else None
    if profile['robot']:
        from kinematics import joint_frames_from_usd
        if UsdGeom.GetStageMetersPerUnit(Usd.Stage.Open(str(args.robot_usd)))!=1.:
            raise ValueError('Kinematic audit requires robot joint-frame distances in metres')
        model=dict(asset_sha256=profile['robot']['asset_sha256'],
                   chain=joint_frames_from_usd(Usd.Stage.Open(str(args.robot_usd)),UsdPhysics))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        (args.output.parent/'robot_kinematics.json').write_text(json.dumps(model,indent=2)+'\n')
    profile['candidates'] = [candidate(profile,vertex=i) for i in range(3)]+[candidate(profile,mode='wall')]
    profile['candidate_preflight'] = [preflight(profile,c) for c in profile['candidates']]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(profile,indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/peg_grasp_v2_matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    outer,inner = contours(mesh,(lo+hi)/2)
    for ax in axes:
        for points in (outer,inner):
            closed = np.vstack([points,points[0]])*1000
            ax.plot(*closed.T,color='black',lw=1.5)
        ax.set_aspect('equal');ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)');ax.grid(alpha=.2)
        ax.add_patch(plt.Circle(center*1000,2,color='grey',alpha=.4,label='Piolo della board'))
    axes[0].set_title('Sezione della mesh e collider v1')
    from matplotlib.patches import Rectangle
    bounds = mesh.bounds*1000
    axes[0].add_patch(Rectangle(bounds[0,:2],*(bounds[1,:2]-bounds[0,:2]),fill=False,color='red',ls='--',label='Contorno collider v1'))
    axes[0].add_patch(plt.Circle(center*1000,3.5,fill=False,color='red',ls='--'))
    axes[0].legend(fontsize=8)
    axes[1].set_title('Candidati v2: vertici e parete del foro')
    for i,c in enumerate(profile['candidates']):
        x,y = np.array(c['tip_local_m'][:2])*1000
        axes[1].scatter(x,y);axes[1].annotate(f'V{i}' if i<3 else 'Parete',(x,y),xytext=(4,4),textcoords='offset points')
        direction = np.array([np.cos(c['yaw_rad']),np.sin(c['yaw_rad'])])*3
        axes[1].plot([x-direction[0],x+direction[0]],[y-direction[1],y+direction[1]],alpha=.6)
    image_path = REPO/'docs/images/peg_grasp_v2_geometry.png'
    image_path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(image_path,dpi=160);plt.close(fig)
    summary={k:profile[k] for k in ('vertices','triangles','bounds_m','sampled_max_radial_error_mm','candidate_preflight')}
    summary['robot']={k:v for k,v in profile['robot'].items() if k!='jaw_geometry_tip_frame'} if profile['robot'] else None
    print(json.dumps(summary,indent=2))
    print(f'Profile: {args.output}; colliders={len(cells)}; figure={image_path}')


if __name__ == '__main__':
    main()
