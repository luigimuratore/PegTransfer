"""Unassisted contact experiment. Import only AFTER AppLauncher."""
import numpy as np
import torch
from pathlib import Path
import hashlib
from pxr import Gf, Usd, UsdGeom, UsdPhysics, PhysxSchema
from scipy.spatial import ConvexHull

import isaaclab.sim as sim
import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.sim.utils import clone
from isaaclab.utils.math import quat_apply_inverse, quat_mul, quat_apply, subtract_frame_transforms

from geometry import PROFILE, load_profile, sectors, verify_asset, surface_label


@configclass
class ContactPegCfg(sim.UsdFileCfg):
    profile_path: str = str(PROFILE)
    mass_kg: float = .005
    friction: float = .6
    contact_stiffness: float = 1000.
    contact_damping: float = .5


@configclass
class ContactRobotCfg(sim.UsdFileCfg):
    finger_friction: float = .9


@clone
def spawn_contact_robot(prim_path,cfg,translation=None,orientation=None,**kwargs):
    root=sim.spawn_from_usd(prim_path,cfg,translation,orientation,**kwargs)
    material_path=f'{prim_path}/GraspV2FingerMaterial'
    material=sim.RigidBodyMaterialCfg(static_friction=cfg.finger_friction,
        dynamic_friction=cfg.finger_friction,restitution=0.,friction_combine_mode='min')
    material.func(material_path,material)
    for finger in (1,2):
        sim.bind_physics_material(f'{prim_path}/psm_tool_gripper{finger}_link',
                                 material_path,stage=root.GetStage())
    return root


def material_evidence(env,friction):
    """Read cooked PhysX materials, not just intended USD attributes."""
    evidence={}
    for arm in (1,2):
        asset=env.scene[f'robot_{arm}']
        counts=[asset._physics_sim_view.create_rigid_body_view(path).max_shapes
                for path in asset.root_physx_view.link_paths[0]]
        if sum(counts)!=asset.root_physx_view.max_shapes:
            raise RuntimeError('Cannot map finger collision shapes to PhysX materials')
        properties=asset.root_physx_view.get_material_properties()[0].cpu().numpy()
        for finger in (1,2):
            ids,_=asset.find_bodies(f'psm_tool_gripper{finger}_link')
            if len(ids)!=1:
                raise RuntimeError('Cannot identify finger body for material verification')
            index=ids[0];start=sum(counts[:index]);count=counts[index]
            samples=properties[start:start+count]
            if count==0 or not np.allclose(samples[:,:2],friction,atol=1e-5,rtol=0):
                raise RuntimeError(f'PSM{arm} finger{finger}: actual friction differs from requested {friction}; '
                                   f'shapes={count}; coefficients={np.unique(samples[:,:2],axis=0).tolist()}')
            evidence[f'psm{arm}_finger{finger}']=dict(shapes=count,
                static_friction_range=[float(samples[:,0].min()),float(samples[:,0].max())],
                dynamic_friction_range=[float(samples[:,1].min()),float(samples[:,1].max())])
    properties=env.scene['object'].root_physx_view.get_material_properties()[0].cpu().numpy()
    if not np.allclose(properties[:,:2],friction,atol=1e-5,rtol=0):
        raise RuntimeError(f'Peg actual friction differs from requested friction {friction}; '
                           f'shapes={len(properties)}; coefficients={np.unique(properties[:,:2],axis=0).tolist()}')
    evidence['peg']=dict(shapes=len(properties),
        static_friction_range=[float(properties[:,0].min()),float(properties[:,0].max())],
        dynamic_friction_range=[float(properties[:,1].min()),float(properties[:,1].max())])
    return evidence


def initialize_contact_physics(env,friction):
    """Set cooked peg coefficients once, then verify them before the probe.

    USD collisionEnabled=False does not remove a collider's material binding.
    PhysX's material buffer can still contain original/disabled shapes, not
    just the mesh-derived cells. Use the same initialization-time tensor API
    as Isaac Lab's material randomization; preserve the restitution column
    (including its compliant-contact representation).
    """
    view=env.scene['object'].root_physx_view
    materials=view.get_material_properties().clone()
    before=materials[...,:2].cpu().numpy().copy()
    env.grasp_material_initial_evidence=dict(peg_coefficients=np.unique(before.reshape(-1,2),axis=0).tolist(),
                                           shape_slots=int(materials.shape[1]))
    if not np.isfinite(before).all():
        raise RuntimeError('Peg PhysX material buffer contains non-finite friction')
    needs_assignment=not np.allclose(before,friction,atol=1e-5,rtol=0)
    if needs_assignment:
        materials[...,:2]=friction
        indices=torch.arange(view.count,dtype=torch.int32,device='cpu')
        view.set_material_properties(materials,indices)
    env.grasp_material_evidence=material_evidence(env,friction)
    env.grasp_material_evidence['peg']['initial_coefficients']=env.grasp_material_initial_evidence['peg_coefficients']
    env.grasp_material_evidence['peg']['initialization_method']=(
        'USD binding plus PhysX material tensor assignment' if needs_assignment else 'USD binding already correct in PhysX')
    print(f'[GRASP V2] PhysX friction verified on peg and four fingers: {friction:g}; '
          f'peg shape slots={materials.shape[1]}',flush=True)
    for arm in (1,2):
        for finger in (1,2):
            sensor=env.scene[f'contact_{arm}_{finger}']
            if sensor.data.force_matrix_w is None or sensor.data.force_matrix_w.shape[2]!=1:
                raise RuntimeError('Finger-to-peg contact filter did not initialize correctly')


@clone
def spawn_contact_peg(prim_path,cfg,translation=None,orientation=None,**kwargs):
    profile = load_profile(cfg.profile_path)
    verify_asset(profile,cfg.usd_path)
    if tuple(cfg.scale) != (profile['scale'],)*3:
        raise ValueError('Collider profile and visual scale must agree')
    visual = cfg.copy()
    visual.collision_props.collision_enabled = False
    root = sim.spawn_from_usd(prim_path,visual,translation,orientation,**kwargs)
    stage = root.GetStage()
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
    material_path = f'{prim_path}/ContactMaterial'
    material = sim.RigidBodyMaterialCfg(static_friction=cfg.friction,dynamic_friction=cfg.friction,
        restitution=0.,friction_combine_mode='min',compliant_contact_stiffness=cfg.contact_stiffness,
        compliant_contact_damping=cfg.contact_damping)
    material.func(material_path,material)
    for i,points in enumerate(sectors(profile)):
        hull = ConvexHull(points)
        mesh = UsdGeom.Mesh.Define(stage,f'{prim_path}/ContactCell_{i:04d}')
        mesh.CreatePointsAttr([Gf.Vec3f(*(p/profile['scale'])) for p in points])
        mesh.CreateFaceVertexCountsAttr([3]*len(hull.simplices))
        faces = []
        for tri,normal in zip(hull.simplices,hull.equations[:,:3]):
            a,b,c = points[tri]
            if np.dot(np.cross(b-a,c-a),normal) < 0:
                tri = tri[::-1]
            faces.extend(int(v) for v in tri)
        mesh.CreateFaceVertexIndicesAttr(faces)
        mesh.CreateSubdivisionSchemeAttr('none');mesh.CreatePurposeAttr('guide')
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
        contact = PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
        contact.CreateContactOffsetAttr(.0001);contact.CreateRestOffsetAttr(0.)
        sim.bind_physics_material(str(mesh.GetPath()),material_path,stage=stage)
    # Bind the inherited disabled collider as well: disabling collision does
    # not delete its original material from the USD or cooked shape buffers.
    sim.bind_physics_material(prim_path,material_path,stage=stage)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(cfg.mass_kg)
    return root


@configclass
class Events:
    reset_scene = EventTermCfg(func=base_mdp.reset_scene_to_default,mode='reset')


def observation(env):
    return env.scene['object'].data.root_pos_w-env.scene.env_origins


@configclass
class Observations:
    @configclass
    class Policy(ObservationGroupCfg):
        position = ObservationTermCfg(func=observation)
        enable_corruption = False
        concatenate_terms = True
    policy: Policy = Policy()


def make_env(args,profile):
    # Reuse only scene/IK definitions. No legacy events, rewards, env.step or attachment logic.
    from robotic.surgery.tasks.surgical.peg_transfer.env_cfg import PegTransferEnvCfg
    from robotic.surgery.tasks.surgical.peg_transfer.sac_actions import ContinuousJawAction, ContinuousJawActionCfg

    class ContactJawAction(ContinuousJawAction):
        def process_actions(self,actions):
            self._raw[:] = actions.clamp(-1,1)
            self._targets[:] = .5*(self._raw+1)/2*self.signs

    scene_cfg = PegTransferEnvCfg()
    scene_cfg.scene.num_envs = 1
    bottom = profile['bounds_m'][0][2]
    scene_cfg.scene.object.init_state.pos = (-profile['hole_center_m'][0],0.,.010-bottom+.0004)
    spawn = ContactPegCfg(usd_path=scene_cfg.scene.object.spawn.usd_path,scale=(profile['scale'],)*3,
        func=spawn_contact_peg,mass_kg=args.mass_g/1000,friction=args.friction,
        contact_stiffness=args.contact_stiffness,contact_damping=args.contact_damping,
        rigid_props=sim.RigidBodyPropertiesCfg(disable_gravity=False,max_linear_velocity=2.,
            max_angular_velocity=20.,max_depenetration_velocity=.2,
            solver_position_iteration_count=32,solver_velocity_iteration_count=8),
        collision_props=sim.CollisionPropertiesCfg(collision_enabled=True,contact_offset=.0001,rest_offset=0.))
    scene_cfg.scene.object.spawn = spawn
    local_robot = args.robot_usd
    if local_robot is None:
        choices = [Path(__file__).resolve().parents[2]/'data/Isaac/Healthcare/0.5.0/132c82d/Robots/dVRK/PSM/psm.usd',
            Path.home()/'.cache/ov/client/https/d36f3a4d2205f4288e3dfc1961a891b4254884432f41d7822f54437aca9b2d45.usd']
        local_robot = next((path for path in choices if path.is_file()),None)
    if profile['robot'] is None or local_robot is None:
        raise ValueError('Inspect a local PSM USD with inspect --robot_usd FILE; use probe --robot_usd FILE')
    if hashlib.sha256(local_robot.read_bytes()).hexdigest()!=profile['robot']['asset_sha256']:
        raise ValueError('PSM USD changed: inspect the robot geometry again before probing')
    for arm in (1,2):
        robot = getattr(scene_cfg.scene,f'robot_{arm}')
        # Preserve all original spawn properties, then bind finger materials
        # while spawning, BEFORE the GPU physics scene is initialized.
        contact_spawn=ContactRobotCfg()
        for key,value in vars(robot.spawn.copy()).items():
            setattr(contact_spawn,key,value)
        contact_spawn.func=spawn_contact_robot
        contact_spawn.finger_friction=args.friction
        robot.spawn=contact_spawn
        robot.spawn.usd_path = str(local_robot)
        robot.spawn.activate_contact_sensors = True
        robot.spawn.articulation_props.solver_position_iteration_count = 32
        robot.spawn.articulation_props.solver_velocity_iteration_count = 8
        # These are explicit simulator velocity limits, not the ignored legacy field.
        body = robot.actuators['psm']
        body.velocity_limit = None
        body.velocity_limit_sim = {name: .04 if 'insertion' in name else 1.5
                                   for name in body.joint_names_expr}
        body.effort_limit = None; body.effort_limit_sim = 12.
        action_cfg = getattr(scene_cfg.actions,f'body_{arm}_joint_pos')
        action_cfg.controller.use_relative_mode = False
        action_cfg.scale = 1.
        jaw = robot.actuators['psm_tool']
        jaw.effort_limit = None; jaw.effort_limit_sim = args.jaw_torque
        jaw.velocity_limit = None; jaw.velocity_limit_sim = .25
        jaw.stiffness = args.jaw_stiffness; jaw.damping = .02
        setattr(scene_cfg.actions,f'finger_{arm}_joint_pos',
            ContinuousJawActionCfg(asset_name=f'robot_{arm}',class_type=ContactJawAction))
        for finger in (1,2):
            sensor = ContactSensorCfg(prim_path=f'{{ENV_REGEX_NS}}/Robot_{arm}/psm_tool_gripper{finger}_link',
                update_period=0.,history_length=1,track_contact_points=True,
                max_contact_data_count_per_prim=256,filter_prim_paths_expr=['{ENV_REGEX_NS}/Object'])
            setattr(scene_cfg.scene,f'contact_{arm}_{finger}',sensor)
    cfg = ManagerBasedEnvCfg(scene=scene_cfg.scene,actions=scene_cfg.actions,
                            observations=Observations(),events=Events(),decimation=4,seed=args.seed)
    cfg.sim.device = args.device;cfg.sim.dt = .0025;cfg.sim.render_interval = 16
    cfg.sim.physx.enable_ccd = False
    cfg.viewer.eye = (.11,-.11,.13);cfg.viewer.lookat = (0.,0.,.035)
    env = ManagerBasedEnv(cfg)
    # Ownership is returned before verification so main's finally can close
    # the environment even if contact initialization raises an exception.
    return env


def control(env,target_pos_w,target_quat_w,jaw_angle,arm):
    # Absolute 7D poses plus one jaw action per arm. Only the active arm follows
    # the bounded trajectory; hold the other arm at its measured pose, jaws open.
    action = torch.zeros((1,16),device=env.device)
    for index in (1,2):
        robot = env.scene[f'robot_{index}']
        frame = env.scene[f'ee_{index}_frame'].data
        pos = target_pos_w if index==arm else frame.target_pos_w[:,0]
        quat = target_quat_w if index==arm else frame.target_quat_w[:,0]
        local_pos,local_quat = subtract_frame_transforms(
            robot.data.root_pos_w,robot.data.root_quat_w,pos,quat)
        offset = (index-1)*8
        action[:,offset:offset+7] = torch.cat([local_pos,local_quat],dim=1)
        action[:,offset+7] = 4*jaw_angle-1 if index==arm else 1.
    return action


def telemetry(env,profile,arm):
    obj = env.scene['object'].data
    frame = env.scene[f'ee_{arm}_frame'].data
    pos,quat = obj.root_pos_w,obj.root_quat_w
    lo,hi = np.asarray(profile['bounds_m'])
    corners = torch.tensor([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1])
                            for z in (lo[2],hi[2])],device=env.device,dtype=pos.dtype)
    world = quat_apply(quat.expand(8,-1),corners)+pos
    tip,toolq = frame.target_pos_w[:,0],frame.target_quat_w[:,0]
    relative = quat_apply_inverse(toolq,pos-tip)[0]
    relative_quat = quat_mul(torch.cat([toolq[:,:1],-toolq[:,1:]],dim=1),quat)[0]
    vectors,points,labels = [],[],[]
    for finger in (1,2):
        data = env.scene[f'contact_{arm}_{finger}'].data
        vectors.append(data.force_matrix_w[0,0,0].cpu().tolist())
        p = data.contact_pos_w[0,0,0].cpu().tolist()
        points.append(p if np.isfinite(p).all() else None)
        if points[-1] is not None:
            local = quat_apply_inverse(quat,torch.tensor([p],device=env.device,dtype=pos.dtype)-pos)[0]
            labels.append(surface_label(profile,local.cpu().numpy()))
        else:
            labels.append('none')
    robot = env.scene[f'robot_{arm}']
    ids,_ = robot.find_joints('psm_tool_gripper.*_joint')
    limits = robot.data.soft_joint_pos_limits[0].cpu().numpy()
    joints = robot.data.joint_pos[0].cpu().numpy()
    margins = np.minimum(joints-limits[:,0],limits[:,1]-joints)
    near_limits = [name for i,name in enumerate(robot.joint_names) if 'gripper' not in name
                   and margins[i] < (.0005 if 'insertion' in name else .035)]
    return dict(peg_root_w_m=pos[0].cpu().tolist(),tool_tip_w_m=tip[0].cpu().tolist(),
        tool_quaternion_wxyz=toolq[0].cpu().tolist(),
        joint_positions=dict(zip(robot.joint_names,joints.tolist())),
        joint_velocities=dict(zip(robot.joint_names,robot.data.joint_vel[0].cpu().tolist())),
        joints_near_limit=near_limits,
        peg_in_tool_m=relative.cpu().tolist(),peg_quat_in_tool=relative_quat.cpu().tolist(),
        peg_bottom_m=float(world[:,2].min()),peg_speed_m_s=float(obj.root_lin_vel_w[0].norm()),
        jaw_rad=env.scene[f'robot_{arm}'].data.joint_pos[0,ids].abs().cpu().tolist(),
        finger_force_vectors_n=vectors,finger_contact_points_w_m=points,finger_contact_surfaces=labels,
        finger_normal_force_n=np.linalg.norm(vectors,axis=1).tolist())
