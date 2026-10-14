"""Mesh-derived grasp geometry. Metres, peg-root frame; no Isaac imports."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PROFILE = Path(__file__).with_name('peg_profile.json')


def load_profile(path=PROFILE):
    return json.loads(Path(path).read_text())


def verify_asset(profile, path):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != profile['asset_sha256']:
        raise ValueError('Peg USD differs from the inspected mesh: run inspect again')


def ray_radius(polygon, center, angle):
    """First positive ray/closed-contour intersection, rejecting non-star-shaped holes."""
    p = np.asarray(polygon, dtype=float)[:, :2] - center
    q = np.roll(p, -1, axis=0)
    d = np.array([math.cos(angle), math.sin(angle)])
    cross = lambda a, b: a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]
    edges = q-p
    denominator = cross(d, edges)
    good = np.abs(denominator) > 1e-12
    t = np.full(len(p), np.nan)
    u = t.copy()
    t[good] = cross(p[good], edges[good]) / denominator[good]
    u[good] = cross(p[good], d) / denominator[good]
    hits = np.unique(np.round(t[(t > 0) & (u >= -1e-8) & (u <= 1+1e-8)], 10))
    if len(hits) != 1:
        raise ValueError(f'Contour is not star-shaped at angle={angle}: {hits}')
    return float(hits[0])


def radii_at(profile, z):
    levels = profile['levels']
    zs = np.array([level['z_m'] for level in levels])
    return tuple(np.array([np.interp(z, zs, [level[key][i] for level in levels])
                          for i in range(len(profile['angles_rad']))])
                 for key in ('inner_radius_m', 'outer_radius_m'))


def sectors(profile):
    """Layered convex cells follow outer triangle AND hole, including edge rounding."""
    angles = np.asarray(profile['angles_rad'])
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    center = np.asarray(profile['hole_center_m'])
    cells = []
    for low, high in zip(profile['levels'], profile['levels'][1:]):
        for i in range(len(angles)):
            j = (i+1) % len(angles)
            points = []
            for level in (low, high):
                inner, outer = np.asarray(level['inner_radius_m']), np.asarray(level['outer_radius_m'])
                xy = center + directions[[i,i,j,j]] * np.array([inner[i],outer[i],outer[j],inner[j]])[:,None]
                points.extend(np.column_stack([xy, np.full(4, level['z_m'])]))
            cells.append(np.asarray(points))
    return cells


def candidate(profile, mode='vertex', vertex=0, height=.5):
    """Initial candidates only; collision/force validation must happen in simulation."""
    if not 0 <= vertex < 3 or not .1 <= height <= .9:
        raise ValueError('vertex must be 0..2; height fraction must be 0.1..0.9')
    lo, hi = np.asarray(profile['bounds_m'])[:,2]
    # Tool tip is the lowest jaw point at the upright, closed reference pose.
    z = lo+(hi-lo)*height if mode == 'vertex' else hi-.001
    theta = vertex*2*math.pi/3 if mode == 'vertex' else math.pi
    inner, outer = radii_at(profile, z)
    angles = np.asarray(profile['angles_rad'])
    ri, ro = np.interp(theta, angles, inner), np.interp(theta, angles, outer)
    distance = ro-.002 if mode == 'vertex' else (ri+ro)/2
    center = np.asarray(profile['hole_center_m'])
    xy = center+distance*np.array([math.cos(theta),math.sin(theta)])
    # The two wall pads exchange inner/outer roles under a 180-degree roll.
    # Use the upright zero-roll branch instead of forcing a half turn of the wrist.
    yaw = theta+math.pi/2 if mode == 'vertex' else 0.
    from scipy.spatial.transform import Rotation
    if mode=='vertex':
        # Pads extend along local +Z toward the hinge. Put the hinge outside the triangle,
        # jaw closing axis tangent to the vertex and jaw breadth vertical.
        x=np.array([-math.sin(theta),math.cos(theta),0.])
        z_axis=np.array([math.cos(theta),math.sin(theta),0.])
        matrix=np.column_stack([x,np.cross(z_axis,x),z_axis])
        quaternion=Rotation.from_matrix(matrix).as_quat()[[3,0,1,2]].tolist()
        approach=z_axis.tolist()
    else:
        quaternion=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)]
        approach=[0.,0.,1.]
    return dict(mode=mode, vertex=vertex, height_fraction=height,
                tip_local_m=[*xy, z], yaw_rad=yaw,
                tool_quaternion_local_wxyz=quaternion,approach_direction_local=approach,
                approach='side_entry' if mode=='vertex' else 'top_entry',
                open_angle_rad=.45 if mode == 'vertex' else .15,
                wall_thickness_m=float(ro-ri) if mode == 'wall' else None)


def jaw_points(profile,plan):
    from scipy.spatial.transform import Rotation
    points=[]
    orientation=Rotation.from_quat(np.asarray(plan['tool_quaternion_local_wxyz'])[[1,2,3,0]])
    for pad,sign in zip(profile['robot']['jaw_geometry_tip_frame'],(-1,1)):
        pivot=np.asarray(pad['pivot_m'])
        rotation=Rotation.from_rotvec(np.asarray(pad['joint_axis'])*sign*plan['open_angle_rad'])
        pts=rotation.apply(np.asarray(pad['points_m'])-pivot)+pivot
        points.append(orientation.apply(pts)+plan['tip_local_m'])
    return points


def contact_candidate(profile,mode,vertex,height,arm):
    """Oblique entry preserves contact sites while accommodating the measured wrist limits."""
    from scipy.spatial.transform import Rotation
    plan=candidate(profile,mode,vertex,height)
    if mode=='wall':
        # PSM1 uses the left wall, PSM2 the upper-right wall. Both enter from
        # above/outside; the inward finger stays clear of the source post.
        theta=math.pi if arm==1 else math.pi/3
        z=plan['tip_local_m'][2]
        inner,outer=radii_at(profile,z)
        ri=np.interp(theta,profile['angles_rad'],inner)
        ro=np.interp(theta,profile['angles_rad'],outer)
        xy=np.asarray(profile['hole_center_m'])+(ri+ro)/2*np.array([math.cos(theta),math.sin(theta)])
        plan['tip_local_m']=[*xy,z]
        tilt=math.radians(20.)
        x=np.array([math.cos(theta)*math.cos(tilt),math.sin(theta)*math.cos(tilt),-math.sin(tilt)])
        z_axis=np.array([math.cos(theta)*math.sin(tilt),math.sin(theta)*math.sin(tilt),math.cos(tilt)])
        orientation=Rotation.from_matrix(np.column_stack([x,np.cross(z_axis,x),z_axis]))
        if arm==1:
            # Equivalent pad exchange avoids a needless 180-degree wrist roll.
            orientation=orientation*Rotation.from_rotvec([0,0,math.pi])
        plan.update(wall_angle_deg=math.degrees(theta),wall_thickness_m=float(ro-ri),
                    entry_tilt_deg=20.,pinch_twist_deg=0.,yaw_rad=0. if arm==1 else theta)
    else:
        theta=vertex*2*math.pi/3
        tilt=math.radians(30.)
        twist=math.radians(25. if arm==1 else 0.)
        x=np.array([-math.sin(theta),math.cos(theta),0.])
        z_axis=np.array([math.cos(theta)*math.cos(tilt),math.sin(theta)*math.cos(tilt),math.sin(tilt)])
        orientation=Rotation.from_matrix(np.column_stack([x,np.cross(z_axis,x),z_axis]))
        orientation=orientation*Rotation.from_rotvec([0,0,twist])
        plan.update(entry_tilt_deg=30.,pinch_twist_deg=math.degrees(twist))
    plan.update(tool_quaternion_local_wxyz=orientation.as_quat()[[3,0,1,2]].tolist(),
                approach_direction_local=z_axis.tolist(),approach='oblique_entry',arm=arm)
    if mode=='vertex':
        # Enter 1 mm deeper on the two flanks, with enough opening to keep
        # both pads clear. Compensate the tilted pad geometry before contact.
        plan['tip_local_m']=(np.asarray(plan['tip_local_m'])-
                             .001*np.array([math.cos(theta),math.sin(theta),0.])).tolist()
        plan['open_angle_rad']=.5
        plan['radial_inset_m']=.003
        plan=balance_vertex_candidate(profile,plan)
    return plan


def closure_contact_screen(profile,plan,angle_step=.005):
    """First sampled pad/peg overlap as coupled jaws close; not contact dynamics."""
    trial=plan.copy();touch=[None,None]
    angles=np.linspace(plan['open_angle_rad'],0.,math.ceil(plan['open_angle_rad']/angle_step)+1)
    for angle in angles:
        trial['open_angle_rad']=float(angle)
        for i,points in enumerate(jaw_points(profile,trial)):
            if touch[i] is None and occupied(profile,points).any():
                touch[i]=float(angle)
        if all(value is not None for value in touch):
            break
    return dict(first_overlap_angle_rad=touch,
                angle_imbalance_rad=abs(touch[0]-touch[1]) if None not in touch else None,
                note='Sampled undeformed pad/peg geometry, not a force or friction proof')


def balance_vertex_candidate(profile,plan):
    """Center coupled pads on the two flanks, without independent jaw control."""
    from scipy.spatial.transform import Rotation
    axis=Rotation.from_quat(np.asarray(plan['tool_quaternion_local_wxyz'])[[1,2,3,0]]).as_matrix()[:,0]
    tip=np.asarray(plan['tip_local_m']);choices=[]
    for offset in np.linspace(-.001,.001,21):
        trial=plan.copy();trial['tip_local_m']=(tip+offset*axis).tolist()
        if not preflight(profile,trial)['sampled_arrival_clear']:
            continue
        screen=closure_contact_screen(profile,trial)
        if screen['angle_imbalance_rad'] is not None:
            choices.append((screen['angle_imbalance_rad'],abs(float(offset)),float(offset),trial,screen))
    if not choices:
        raise ValueError('No open, two-flank vertex candidate in the sampled geometry')
    _,_,offset,trial,screen=min(choices,key=lambda item:item[:2])
    trial.update(pinch_center_offset_m=offset,closure_geometry_check=screen)
    return trial


def occupied(profile,points):
    """Vectorized sampled radial profile occupancy; not a swept collision proof."""
    points=np.asarray(points)
    z=points[:,2]
    levels=profile['levels'];zs=np.asarray([l['z_m'] for l in levels])
    lower=np.searchsorted(zs,z,side='right').clip(1,len(zs)-1)-1
    frac=((z-zs[lower])/(zs[lower+1]-zs[lower])).clip(0,1)
    xy=points[:,:2]-profile['hole_center_m']
    angle=np.arctan2(xy[:,1],xy[:,0])%(2*math.pi)
    n=len(profile['angles_rad']);ai=angle/(2*math.pi)*n
    i=np.floor(ai).astype(int)%n;j=(i+1)%n;f=ai-np.floor(ai)
    bounds=[]
    for key in ('inner_radius_m','outer_radius_m'):
        radii=np.asarray([l[key] for l in levels])
        bottom=radii[lower,i]*(1-f)+radii[lower,j]*f
        top=radii[lower+1,i]*(1-f)+radii[lower+1,j]*f
        bounds.append(bottom*(1-frac)+top*frac)
    radius=np.linalg.norm(xy,axis=1)
    return (z>=zs[0]) & (z<=zs[-1]) & (radius>bounds[0]) & (radius<bounds[1])


def preflight(profile,plan):
    """Screen sampled jaw geometry at arrival against peg, source post and board."""
    if profile['robot'] is None:
        return dict(checked=False)
    pads=jaw_points(profile,plan)
    counts=[int(occupied(profile,pts).sum()) for pts in pads]
    post_counts=[]
    for pts in pads:
        # Source post is at the hole center; board = peg min-z at rest.
        distance=np.linalg.norm(pts[:,:2]-profile['hole_center_m'],axis=1)
        world_z=pts[:,2]+.010-profile['bounds_m'][0][2]
        post_counts.append(int(((distance<.0021)&(world_z<.0356)&(world_z>.0104)).sum()))
    board_margin=min(pts[:,2].min()-profile['bounds_m'][0][2] for pts in pads)
    return dict(checked=True,jaw_surface_samples_inside_peg=counts,
                jaw_surface_samples_inside_source_post=post_counts,
                minimum_jaw_board_clearance_mm=float(board_margin*1000),
                sampled_arrival_clear=bool(not any(counts+post_counts) and board_margin>.0002),
                note='Sampled jaw surfaces at arrival only; shaft and full approach require physics validation')


def physical_hold(row, initial_bottom, previous_relative, force_threshold=.01):
    """Contact alone or peg height alone cannot certify a sustained grasp."""
    forces = np.asarray(row['finger_force_vectors_n'])
    norms = np.linalg.norm(forces,axis=1)
    opposed = np.dot(forces[0],forces[1]) < -.25*np.prod(norms)
    relative = np.asarray(row['peg_in_tool_m'])
    slip = np.linalg.norm(relative-np.asarray(previous_relative))
    return bool(np.all(norms > force_threshold) and opposed and
                row['peg_bottom_m'] > max(.0375,initial_bottom+.020) and
                row['peg_speed_m_s'] < .02 and slip < .0015)


def surface_label(profile, point_local):
    """Classify measured finger contact on inner wall, outer wall, or rounded lip."""
    x,y,z = np.asarray(point_local)
    lo,hi = np.asarray(profile['bounds_m'])[:,2]
    if z < lo-.0004 or z > hi+.0004:
        return 'unknown'
    xy = np.array([x,y])-profile['hole_center_m']
    angle = math.atan2(xy[1],xy[0]) % (2*math.pi)
    inner,outer = radii_at(profile,z)
    angles = [*profile['angles_rad'],2*math.pi]
    ri = np.interp(angle,angles,np.append(inner,inner[0]))
    ro = np.interp(angle,angles,np.append(outer,outer[0]))
    radius = np.linalg.norm(xy)
    distances = [abs(radius-ri),abs(radius-ro)]
    if min(distances)<.0004:
        return ('inner','outer')[int(distances[1]<distances[0])]
    return 'lip' if min(abs(z-lo),abs(z-hi))<.0004 else 'unknown'


def valid_contact_pair(row,mode):
    points = row.get('finger_contact_points_w_m',[])
    if len(points)!=2 or any(p is None for p in points):
        return False
    if np.linalg.norm(np.asarray(points[0])-points[1]) < .0005:
        return False
    labels = sorted(row.get('finger_contact_surfaces',[]))
    return labels == (['inner','outer'] if mode=='wall' else ['outer','outer'])
