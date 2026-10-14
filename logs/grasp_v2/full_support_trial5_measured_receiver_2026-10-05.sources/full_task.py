"""Unassisted two-PSM transfer. Wall contacts, measured handover and R2 placement.

This is a teacher experiment, not a newly trained full-task policy.
An optional saved motor policy controls the already validated donor pickup.
Both arms receive explicit pose/jaw commands; no object commands or attachments.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import importlib.metadata
import numpy as np

from geometry import load_profile,contact_candidate,valid_contact_pair,preflight
from grip import GripController
from handover_grip import WallReceiverGrip
from holding_compliance import HoldingCompliance
from motion import PoseTrajectory,MotionWatchdog,pose_reached,rotation
from full_geometry import TRANSFER_XY,TARGET_XY,wall_pose,approach_points,placement_evidence,screen_full_route,transfer_wall_plan
from learning import MotorPolicy,observation,motor_command,PHYSICS_FILES

REPO=Path(__file__).resolve().parents[2]
STAGES=('settle','donor_above','donor_outside','donor_reach','donor_close','donor_lift','donor_hold',
        'donor_transfer','receiver_above','receiver_outside','receiver_reach','receiver_close',
        'donor_release','donor_retreat','receiver_hold','receiver_transport','receiver_insert',
        'receiver_release','receiver_retreat','placement_settle')


def secure_contact(row,target_force):
    f=np.asarray(row['finger_normal_force_n']);v=np.asarray(row['finger_force_vectors_n'])
    return bool((f>=target_force-.05).all() and np.dot(*v)<-.25*np.prod(f) and valid_contact_pair(row,'wall'))


def fixture_holds(support,arm):
    return bool(support and support.evidence()['type']=='temporary_physx_fixed_joint'
                and support.holder==arm and not support.break_events)


def two_arm_action(env,positions,quaternions,jaws):
    import torch
    from isaaclab.utils.math import subtract_frame_transforms
    action=torch.empty((1,16),device=env.device)
    for index in (0,1):
        robot=env.scene[f'robot_{index+1}']
        pos=torch.as_tensor(positions[index][None],device=env.device,dtype=robot.data.root_pos_w.dtype)
        quat=torch.as_tensor(quaternions[index][None],device=env.device,dtype=robot.data.root_quat_w.dtype)
        p,q=subtract_frame_transforms(robot.data.root_pos_w,robot.data.root_quat_w,pos,quat)
        action[:,index*8:index*8+7]=torch.cat([p,q],dim=1)
        action[:,index*8+7]=4*jaws[index]-1.
    return action


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--stage_steps',type=int,default=6000)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--enable_cameras',action='store_true')
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--realtime',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--donor_policy',type=Path)
    p.add_argument('--video_path',type=Path)
    p.add_argument('--robot_usd',type=Path)
    p.add_argument('--camera_eye',type=float,nargs=3,default=(.025,.32,.25))
    p.add_argument('--camera_target',type=float,nargs=3,default=(.025,0.,.035))
    p.add_argument('--mass_g',type=float,default=5.);p.add_argument('--friction',type=float,default=.9)
    p.add_argument('--contact_stiffness',type=float,default=1000.);p.add_argument('--contact_damping',type=float,default=.5)
    p.add_argument('--jaw_torque',type=float,default=.015);p.add_argument('--jaw_stiffness',type=float,default=.6)
    p.add_argument('--grip_force',type=float,default=.35);p.add_argument('--max_force',type=float,default=1.)
    p.add_argument('--preload_seconds',type=float,default=.4)
    p.add_argument('--free_speed',type=float,default=.025);p.add_argument('--contact_speed',type=float,default=.005)
    p.add_argument('--lift_speed',type=float,default=.003);p.add_argument('--rotation_speed',type=float,default=1.)
    p.add_argument('--donor_depth_mm',type=float,default=1.)
    p.add_argument('--receiver_depth_mm',type=float,default=1.)
    p.add_argument('--donor_open',type=float,default=.15)
    p.add_argument('--receiver_open',type=float,default=.15)
    p.add_argument('--donor_compliance',action='store_true')
    p.add_argument('--donor_balance',action='store_true',help='Experimental balanced closure for a deeper donor wall grasp')
    p.add_argument('--peg_support',type=float,default=0.,help='Optional curriculum spring strength 0..1; 0 is unassisted')
    p.add_argument('--peg_support_type',choices=('spring','fixed'),default='spring')
    args=p.parse_args()
    if args.episodes<1 or args.stage_steps<200:p.error('Positive episodes and stage_steps >= 200 required')
    if not np.isfinite(args.peg_support) or not 0<=args.peg_support<=1:p.error('peg_support must be 0..1')
    if args.peg_support_type=='fixed' and args.peg_support not in (0.,1.):p.error('Ideal fixed support is binary: use 0 or 1')
    if args.video_path:
        if args.episodes!=1:p.error('Record one episode')
        if args.video_path.exists():p.error('Video already exists')
        args.enable_cameras=True
    args.output=args.output or REPO/'logs/grasp_v2'/f'full_wall_{datetime.now():%Y-%m-%d_%H-%M-%S-%f}.json'
    if args.output.exists() or args.output.with_suffix('.trace.jsonl').exists():p.error('Report already exists')
    if not all(.5<=v<=4. for v in (args.donor_depth_mm,args.receiver_depth_mm)):p.error('Depth must be .5..4 mm')
    if not all(.1<=v<=.5 for v in (args.donor_open,args.receiver_open)):p.error('Opening must be .1.. .5 rad')
    profile=load_profile();plans=[transfer_wall_plan(profile,1,args.donor_depth_mm,args.donor_open),
                                transfer_wall_plan(profile,2,args.receiver_depth_mm,args.receiver_open)]
    arrivals=[preflight(profile,plan) for plan in plans]
    if not arrivals[0]['sampled_arrival_clear']:p.error(f'Donor open arrival intersects geometry: {arrivals[0]}')
    if any(arrivals[1]['jaw_surface_samples_inside_peg']):p.error(f'Receiver arrival intersects peg: {arrivals[1]}')
    screen=screen_full_route(profile,plans=plans)
    if not screen['reachable']:p.error(f'Full route fails unchanged joint limits: {screen}')
    policy=MotorPolicy(args.donor_policy,device=args.device) if args.donor_policy else None
    if policy:
        if args.donor_balance:p.error('Balanced donor closure requires a new demonstration; do not alter saved donor policy inference')
        from replay import checkpoint_sources
        snapshot,_=checkpoint_sources(args.donor_policy.resolve())
        for filename in PHYSICS_FILES:
            if (snapshot/filename).read_bytes()!=Path(__file__).with_name(filename).read_bytes():
                p.error(f'Donor physical sources changed: {filename}; keep original physics or retrain')
        original=policy.metadata['experiment']
        for key in ('mass_g','friction','contact_stiffness','contact_damping','jaw_torque','jaw_stiffness',
                    'grip_force','preload_seconds','max_force','free_speed','contact_speed','lift_speed','rotation_speed'):
            if original['parameters'][key]!=getattr(args,key):p.error(f'Donor policy parameter mismatch: {key}')
        if original['candidate']!=plans[0]:p.error('Donor policy grasp candidate differs')
    source_names=('full_task.py','full_geometry.py','handover_grip.py','holding_compliance.py','peg_support.py',
                  'geometry.py','runtime.py','motion.py','grip.py','kinematics.py','learning.py',
                  'torch_backend.py','replay.py','peg_profile.json','robot_kinematics.json')
    source_blobs={name:Path(__file__).with_name(name).read_bytes() for name in source_names}
    source_hashes={name:hashlib.sha256(data).hexdigest() for name,data in source_blobs.items()}
    source_folder=args.output.with_suffix('.sources');source_folder.mkdir(parents=True,exist_ok=False)
    for name,data in source_blobs.items():(source_folder/name).write_bytes(data)
    (source_folder/'source_manifest.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
    sys.path.insert(0,str(REPO/'workflows/robotic_surgery/scripts'))
    from isaaclab.app import AppLauncher
    launcher=AppLauncher(args);app=launcher.app
    env=None;cases=[];error=None;interrupted=False;recorder=None;support=None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    trace_path=args.output.with_suffix('.trace.jsonl')
    try:
        import torch
        torch.set_num_threads(4)
        from runtime import make_env,initialize_contact_physics,telemetry
        env=make_env(args,profile);initialize_contact_physics(env,args.friction)
        if args.peg_support:
            from peg_support import PegSupport,FixedPegSupport
            support=(FixedPegSupport if args.peg_support_type=='fixed' else PegSupport)(env,args.peg_support)
        env.cfg.viewer.eye=tuple(args.camera_eye);env.cfg.viewer.lookat=tuple(args.camera_target)
        if env.viewport_camera_controller is not None:
            env.viewport_camera_controller.update_view_location(eye=args.camera_eye,lookat=args.camera_target)
        else:env.sim.set_camera_view(eye=args.camera_eye,target=args.camera_target)
        if args.video_path:
            from replay import EpisodeVideo
            recorder=EpisodeVideo(env,args.video_path)
            for _ in range(4):env.sim.render()
        def measured(arm):
            row=telemetry(env,profile,arm)
            row['peg_quaternion_wxyz']=env.scene['object'].data.root_quat_w[0].cpu().tolist()
            row['finger_total_force_n']=[float(env.scene[f'contact_{arm}_{f}'].data.net_forces_w[0,0].norm()) for f in (1,2)]
            return row
        with trace_path.open('x') as trace,torch.inference_mode():
            for episode in range(args.episodes):
                if support:support.release()
                env.reset();rows=[measured(1),measured(2)]
                positions=[np.array(r['tool_tip_w_m']) for r in rows]
                quaternions=[np.array(r['tool_quaternion_wxyz']) for r in rows]
                jaws=[plan['open_angle_rad'] for plan in plans]
                grips=[GripController('wall',plan['open_angle_rad'],args.grip_force,args.max_force,args.friction,
                                      args.mass_g/1000,args.preload_seconds) for plan in plans]
                grips[1]=WallReceiverGrip('wall',plans[1]['open_angle_rad'],args.grip_force,args.max_force,args.friction,
                                          args.mass_g/1000,args.preload_seconds)
                if args.donor_balance:
                    grips[0]=WallReceiverGrip('wall',plans[0]['open_angle_rad'],args.grip_force,args.max_force,args.friction,
                                              args.mass_g/1000,args.preload_seconds)
                settled=None;pickup=None;received=None;failure=None;records=[];compliance=None
                for stage in STAGES:
                    rows=[measured(1),measured(2)]
                    if support:
                        if stage=='donor_transfer':support.acquire(1)
                        elif stage=='donor_release':support.acquire(2)
                        elif stage=='receiver_release':support.release()
                    active=0 if stage.startswith('donor') else 1
                    goal=positions[active].copy();goalq=quaternions[active].copy()
                    if stage in ('donor_above','donor_outside','donor_reach'):
                        grasp,goalq=wall_pose(settled['peg_root_w_m'],settled['peg_quaternion_wxyz'],plans[0])
                        above,outside=approach_points(grasp,settled['peg_quaternion_wxyz'],plans[0])
                        goal={'donor_above':above,'donor_outside':outside,'donor_reach':grasp}[stage]
                    elif stage=='donor_close':
                        goal=np.array(rows[0]['tool_tip_w_m']);goalq=np.array(rows[0]['tool_quaternion_wxyz'])
                    elif stage in ('donor_lift','donor_hold'):
                        goal=pickup['position']+[0,0,.035];goalq=pickup['quaternion']
                    elif stage=='donor_transfer':
                        center=np.asarray(rows[0]['peg_root_w_m'])+rotation(rows[0]['peg_quaternion_wxyz']).apply([*profile['hole_center_m'],0.])
                        goal=positions[0]+np.r_[TRANSFER_XY-center[:2],0.]
                    elif stage in ('receiver_above','receiver_outside','receiver_reach'):
                        grasp,goalq=wall_pose(rows[1]['peg_root_w_m'],rows[1]['peg_quaternion_wxyz'],plans[1])
                        above,outside=approach_points(grasp,rows[1]['peg_quaternion_wxyz'],plans[1])
                        goal={'receiver_above':above,'receiver_outside':outside,'receiver_reach':grasp}[stage]
                    elif stage=='receiver_close':
                        goal=np.array(rows[1]['tool_tip_w_m']);goalq=np.array(rows[1]['tool_quaternion_wxyz'])
                    elif stage in ('donor_retreat','receiver_retreat'):
                        goal=np.array(rows[active]['tool_tip_w_m'])+.015*rotation(rows[active]['peg_quaternion_wxyz']).apply(plans[active]['approach_direction_local'])
                    elif stage=='receiver_transport':
                        center=np.asarray(rows[1]['peg_root_w_m'])+rotation(rows[1]['peg_quaternion_wxyz']).apply([*profile['hole_center_m'],0.])
                        goal=positions[1]+np.r_[TARGET_XY-center[:2],0.]
                    elif stage=='receiver_insert':
                        goal=positions[1].copy();goal[2]-=rows[1]['peg_bottom_m']-.010
                    trajectory=PoseTrajectory(rows[active]['tool_tip_w_m'],rows[active]['tool_quaternion_wxyz'])
                    stable=0;achieved=False;watchdog=MotionWatchdog();states=[None,None]
                    release_start=jaws[active]
                    budget=50 if stage=='settle' else args.stage_steps
                    donor_holding=STAGES.index('donor_lift')<=STAGES.index(stage)<=STAGES.index('receiver_close')
                    receiver_holding=STAGES.index('donor_release')<=STAGES.index(stage)<=STAGES.index('receiver_insert')
                    if stage=='receiver_above' and args.donor_compliance:
                        compliance=HoldingCompliance(rows[0],grips[0].reference)
                    contact=stage in ('donor_reach','donor_close','donor_lift','donor_hold','donor_transfer',
                                      'receiver_reach','receiver_close','receiver_hold','receiver_transport','receiver_insert')
                    linear=(args.lift_speed if stage in ('donor_lift','donor_hold','donor_transfer','receiver_hold','receiver_transport')
                            else args.contact_speed if contact else args.free_speed)
                    for step in range(budget):
                        tick=time.monotonic()
                        if not app.is_running():raise KeyboardInterrupt
                        rows=[measured(1),measured(2)];previous_jaws=jaws.copy()
                        previous_reference=trajectory.position.copy();allowed=True;pressure_overrides=[]
                        for arm,holding in enumerate((donor_holding,receiver_holding)):
                            closing=stage==('donor_close' if arm==0 else 'receiver_close')
                            if holding or closing:
                                grips[arm].angle=jaws[arm]
                                states[arm]=grips[arm].update(rows[arm],'close' if closing else 'hold',env.step_dt)
                                jaws[arm]=states[arm]['jaw_target_rad']
                                fixture=fixture_holds(support,arm+1)
                                reason=states[arm]['reason']
                                if reason and fixture and reason in ('finger_contact_lost','grip_pressure_not_recovered'):
                                    pressure_overrides.append(dict(arm=arm+1,reason=reason))
                                elif reason:failure=f'psm{arm+1}_{reason}'
                                if holding:allowed=allowed and (states[arm]['motion_allowed'] or fixture)
                            elif stage==('donor_release' if arm==0 else 'receiver_release'):
                                if max(rows[arm]['finger_normal_force_n'])>=.01 or jaws[arm]<release_start+.02:
                                    jaws[arm]=min(.5,jaws[arm]+.15*env.step_dt)
                            if max(rows[arm]['finger_normal_force_n'])>args.max_force:failure=f'psm{arm+1}_excessive_finger_force'
                        if failure:break
                        if compliance is not None and donor_holding:
                            positions[0],quaternions[0]=compliance.command(rows[0],env.step_dt)
                        if stage=='receiver_close' or (stage=='donor_close' and args.donor_balance):
                            if states[active]['centering_position_w_m'] is not None:
                                goal=np.array(states[active]['centering_position_w_m'])
                            # Centering unloads the receiver's first contact while
                            # the stationary donor restores preload. The slip guard still applies.
                            allowed=True
                        pos,quat=trajectory.advance(rows[active]['tool_tip_w_m'],rows[active]['tool_quaternion_wxyz'],
                            goal,goalq,env.step_dt,linear,.2 if contact else args.rotation_speed,
                            contact=contact,allow_motion=allowed)
                        # Keep the native wall_bc3 donor pickup. Later stages and PSM2
                        # use the physical teacher, pending a separate full-task dataset.
                        if policy and stage.startswith('donor_') and stage.split('_',1)[1] in ('above','outside','reach','close','lift','hold'):
                            name=stage.split('_',1)[1]
                            obs=observation(rows[0],name,goal,goalq,previous_reference,previous_jaws[0])
                            pos,jaws[0]=motor_command(policy.predict(obs),
                                previous_reference if allowed else trajectory.position,rows[0]['tool_tip_w_m'],
                                previous_jaws[0],env.step_dt,linear,contact,allowed)
                            trajectory.position=pos.copy()
                        positions[active]=pos;quaternions[active]=quat
                        env.step(two_arm_action(env,positions,quaternions,jaws))
                        rows=[measured(1),measured(2)]
                        pe=float(np.linalg.norm(goal-rows[active]['tool_tip_w_m'])*1000)
                        re=float((rotation(goalq)*rotation(rows[active]['tool_quaternion_wxyz']).inv()).magnitude())
                        if rows[0]['peg_bottom_m']<.005:failure='peg_dropped'
                        if any(max(r['finger_normal_force_n'])>args.max_force for r in rows):failure='excessive_finger_force'
                        evidence=placement_evidence(rows[1],profile)
                        trace.write(json.dumps(dict(episode=episode,stage=stage,step=step,psm1=rows[0],psm2=rows[1],
                            grip_control=states,jaw_targets_rad=jaws,motion_allowed=allowed,
                            peg_support=support.evidence() if support else None,
                            assisted_pressure_overrides=pressure_overrides,
                            target_error_mm=pe,rotation_error_rad=re,placement=evidence),allow_nan=False)+'\n')
                        if stage=='settle':achieved=step==49
                        elif stage in ('donor_close','receiver_close'):
                            achieved=bool(states[active]['pressure_ready'] and secure_contact(rows[active],args.grip_force))
                        elif stage in ('donor_hold','receiver_hold'):
                            secure=(secure_contact(rows[active],args.grip_force) or fixture_holds(support,active+1)) and rows[active]['peg_bottom_m']-settled['peg_bottom_m']>.033
                            stable=stable+1 if secure and pose_reached(pe,re) else 0
                            achieved=stable>=200
                        elif stage=='donor_release':
                            released=bool(max(rows[0]['finger_normal_force_n'])<.01 and
                                          (secure_contact(rows[1],args.grip_force) or fixture_holds(support,2))
                                          and jaws[0]>=release_start+.02)
                            stable=stable+1 if released else 0;achieved=stable>=25
                        elif stage=='receiver_release':
                            stable=stable+1 if evidence['placed'] and jaws[1]>=release_start+.02 else 0
                            achieved=stable>=25
                        elif stage=='placement_settle':
                            released=bool(max(rows[0]['finger_normal_force_n'])<.01)
                            stable=stable+1 if evidence['placed'] and released else 0;achieved=stable>=100
                        else:
                            stable=stable+1 if pose_reached(pe,re) and allowed else 0;achieved=stable>=10
                            if watchdog.stalled((step+1)*env.step_dt,pe,re) and not achieved and allowed:failure=f'{stage}_motion_stalled'
                        if step==0 or (step+1)%200==0:
                            print(f'[FULL PHYSICAL] {episode+1}/{args.episodes} {stage} step={step+1} '
                                  f'error={pe:.2f}mm rotation={re:.3f}rad '
                                  f'force={[[round(f,3) for f in r["finger_normal_force_n"]] for r in rows]} '
                                  f'allowed={allowed}',flush=True);trace.flush()
                        if args.realtime:time.sleep(max(0,env.step_dt-(time.monotonic()-tick)))
                        if achieved or failure:break
                    records.append(dict(stage=stage,achieved=achieved,steps=step+1,error_mm=pe if step else None,
                                        grip=states,placement=evidence if step else None))
                    print(f'[FULL PHYSICAL] {stage}: achieved={achieved}; reason={failure}',flush=True)
                    if not achieved:failure=failure or f'{stage}_budget_exhausted';break
                    positions[active]=np.array(rows[active]['tool_tip_w_m']);quaternions[active]=np.array(rows[active]['tool_quaternion_wxyz'])
                    if stage=='settle':settled=rows[0]
                    if stage=='donor_close':pickup=dict(position=positions[0].copy(),quaternion=quaternions[0].copy())
                    if stage=='receiver_close':received=rows[1]
                cases.append(dict(episode=episode,full_success=failure is None,reason=failure,stages=records,
                                  peg_support=support.evidence() if support else None,
                                  terminal=rows,placement=placement_evidence(rows[1],profile)))
                if support:support.release()
    except KeyboardInterrupt:interrupted=True
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        import traceback;traceback.print_exc()
    finally:
        report=dict(schema='peg-physical-full-1',source='L5',target='R2',strategy='wall/wall physical teacher with optional learned donor pickup',
                    donor_policy=str(args.donor_policy) if policy else None,full_task_learned_policy=False,
                    assisted_grasp=bool(args.peg_support),object_pose_servo=False,
                    fixed_or_magnetic_attachment=bool(args.peg_support and args.peg_support_type=='fixed'),
                    unassisted_full_success=sum(c['full_success'] for c in cases) if not args.peg_support else 0,
                    peg_support=support.evidence() if support else None,
                    kinematic_route_check=screen,episodes=len(cases),requested_episodes=args.episodes,cases=cases,
                    complete=len(cases)==args.episodes and not interrupted and error is None,
                    full_success=sum(c['full_success'] for c in cases),error=error,interrupted=interrupted,
                    trace=str(trace_path),device=args.device,camera=dict(eye=args.camera_eye,target=args.camera_target),
                    code_sha256=hashlib.sha256(b''.join(source_blobs.values())).hexdigest(),
                    source_snapshot=str(source_folder),source_hashes=source_hashes,
                    donor_policy_sha256=hashlib.sha256(args.donor_policy.read_bytes()).hexdigest() if policy else None,
                    versions={name:importlib.metadata.version(name) for name in ('isaacsim','isaaclab','torch')},
                    parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                    actual_physx_materials=getattr(env,'grasp_material_evidence',None))
        report.update(grasp_plans=plans,arrival_checks=arrivals)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(f'[FULL PHYSICAL] Report: {args.output}; successes={report["full_success"]}/{args.episodes}',flush=True)
        try:
            if env is not None:env.close()
        finally:app.close()
    return int(error is not None or interrupted or len(cases)!=args.episodes or not all(c['full_success'] for c in cases))


if __name__=='__main__':raise SystemExit(main())
