"""User-run contact grasp probe: open approach, pinch, lift, hold, lower, release."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from geometry import PROFILE, contact_candidate, load_profile, physical_hold, valid_contact_pair, preflight
from motion import PoseTrajectory, MotionWatchdog, pose_reached
from kinematics import MODEL, route_check
from grip import grip_controller
from learning import signature,observation,teacher_action,motor_command,save_demonstrations,MotorPolicy

REPO = Path(__file__).resolve().parents[2]


def stage_limits(stage_steps,step_dt):
    """Separate maximum waiting time from the measured grasp hold duration."""
    if stage_steps < 1 or not math.isfinite(step_dt) or step_dt <= 0:
        raise ValueError('Positive stage_steps and finite positive step_dt required')
    names=('above','outside','reach','close','lift','hold','lower','release','retreat')
    return [('settle',math.ceil(.5/step_dt))]+[(name,stage_steps) for name in names]


def hold_steps(step_dt):
    return math.ceil(2./step_dt)


def configure_camera(env,eye,lookat):
    """Front of both PSMs, tilted down towards the operative board."""
    env.cfg.viewer.eye=tuple(eye);env.cfg.viewer.lookat=tuple(lookat)
    if env.viewport_camera_controller is not None:
        env.viewport_camera_controller.update_view_location(eye=eye,lookat=lookat)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('vertex','wall'),default='vertex')
    p.add_argument('--vertex',type=int,choices=(0,1,2),default=None,
                   help='Default: arm-facing vertex (2 for PSM1, 0 for PSM2); explicit choice overrides it')
    p.add_argument('--height',type=float,default=.5,help='Vertex grasp height as fraction of peg height (.1.. .9)')
    p.add_argument('--arm',type=int,choices=(1,2),default=1)
    p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--stage_steps',type=int,default=6000,
                   help='Maximum control steps per phase (6000 = 60 simulated seconds at 100 Hz); exits early when achieved')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--free_speed',type=float,default=.025,help='Free-space reference speed in m/s')
    p.add_argument('--contact_speed',type=float,default=.005,help='Near-peg approach/lowering reference speed in m/s')
    p.add_argument('--lift_speed',type=float,default=.003,help='Pressure-monitored lift reference speed in m/s')
    p.add_argument('--rotation_speed',type=float,default=1.,help='Free-space orientation reference speed in rad/s')
    p.add_argument('--mass_g',type=float,default=5.)
    p.add_argument('--friction',type=float,default=.9,help='Peg/finger friction; experimental, not measured rubber properties')
    p.add_argument('--contact_stiffness',type=float,default=1000.,help='Normal spring N/m; 0=rigid contacts')
    p.add_argument('--contact_damping',type=float,default=.5,help='Normal damping N s/m')
    p.add_argument('--jaw_torque',type=float,default=.015,help='Per-joint actuator effort limit in N m')
    p.add_argument('--jaw_stiffness',type=float,default=.6,help='Jaw drive gain in N m/rad')
    p.add_argument('--grip_force',type=float,default=.35,help='Normal pressure target per finger in N; deadband +/-0.05 N')
    p.add_argument('--preload_seconds',type=float,default=.4,help='Stable measured pressure before lift')
    p.add_argument('--max_force',type=float,default=1.,help='Abort on any measured finger normal force above this N')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--enable_cameras',action='store_true')
    p.add_argument('--realtime',action='store_true',help='Pace the GUI to simulated time')
    p.add_argument('--output',type=Path)
    p.add_argument('--demonstrations',type=Path,help='Record successful wall episodes for offline BC')
    p.add_argument('--policy',type=Path,help='Evaluate learned wall position/pressure gains on CUDA; feedback, orientation and phases remain programmed')
    p.add_argument('--camera_eye',type=float,nargs=3,default=(.025,.32,.25),metavar=('X','Y','Z'))
    p.add_argument('--camera_target',type=float,nargs=3,default=(.025,0.,.035),metavar=('X','Y','Z'))
    p.add_argument('--robot_usd',type=Path,help='Local PSM USD matching the geometry audit')
    args = p.parse_args()
    if (args.demonstrations or args.policy) and args.mode!='wall':
        p.error('First motor training supports wall only; vertex has no verified successful teacher yet')
    if args.demonstrations and args.policy:
        p.error('Collect expert demonstrations separately from learned-policy evaluation')
    if args.demonstrations and args.demonstrations.exists():
        p.error('Demonstrations already exist; choose a new filename')
    if args.episodes < 1 or not .1 <= args.height <= .9:
        p.error('Positive episodes and height .1.. .9 required')
    if args.stage_steps < 1:
        p.error('--stage_steps must be positive')
    for key in ('mass_g','friction','jaw_torque','jaw_stiffness','grip_force','preload_seconds',
                'max_force','free_speed','contact_speed','lift_speed','rotation_speed'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key) <= 0:
            p.error(f'--{key} must be finite and positive')
    for key in ('contact_stiffness','contact_damping'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key) < 0:
            p.error(f'--{key} must be finite and nonnegative')
    args.output = args.output or REPO/'logs/grasp_v2'/f'{args.mode}_psm{args.arm}_{datetime.now():%Y-%m-%d_%H-%M-%S-%f}.json'
    if args.output.exists() or args.output.with_suffix('.trace.jsonl').exists():
        p.error('Output already exists; choose a new filename')
    profile = load_profile()
    if profile['robot'] is None:
        p.error('No inspected robot: run inspect --robot_usd FILE, then probe --robot_usd FILE')
    if args.vertex is None:
        args.vertex = 2 if args.arm==1 else 0
    try:
        plan = contact_candidate(profile,args.mode,args.vertex,args.height,args.arm)
    except ValueError as exc:
        p.error(str(exc))
    try:
        grip_controller(args.mode,plan['open_angle_rad'],args.grip_force,args.max_force,
                       args.friction,args.mass_g/1000,args.preload_seconds)
    except ValueError as exc:
        p.error(str(exc))
    arrival = preflight(profile,plan)
    if arrival.get('checked') and not arrival['sampled_arrival_clear']:
        p.error(f'Open jaw arrival intersects the sampled geometry: {arrival}. Change candidate and inspect it.')
    reachability=route_check(profile,plan,args.free_speed,args.contact_speed,args.rotation_speed)
    if not reachability['reachable']:
        p.error(f'Unreachable approach/lift with unchanged PSM joint limits: {reachability}. '
                'Use the default arm-facing vertex/height; additional timesteps will not fix this pose.')
    experiment=signature(args,profile,plan)
    try:
        policy=MotorPolicy(args.policy,experiment,device=args.device) if args.policy else None
    except (ValueError,OSError,KeyError) as exc:
        p.error(str(exc))
    print(f'[GRASP V2] Pose screened: {plan["entry_tilt_deg"]:g} deg entry tilt; '
          f'{plan["pinch_twist_deg"]:g} deg pinch twist; '
          f'{reachability["samples_checked"]} nominal route samples within unchanged joint limits '
          '(2 deg / 1 mm margin). Contact/dynamics still require this experiment.',flush=True)
    # The simulation namespace must be available before robot/task imports.
    sys.path.insert(0,str(REPO/'workflows/robotic_surgery/scripts'))
    from isaaclab.app import AppLauncher
    print('[GRASP V2] Starting an unassisted contact experiment',flush=True)
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    cases,interrupted,error = [],False,None
    demo_samples=[];demo_result=None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    trace_path = args.output.with_suffix('.trace.jsonl')
    try:
        import torch
        from isaaclab.utils.math import quat_apply,quat_mul,compute_pose_error
        from runtime import make_env,initialize_contact_physics,control,telemetry
        env = make_env(args,profile)
        configure_camera(env,args.camera_eye,args.camera_target)
        initialize_contact_physics(env,args.friction)
        if policy:
            print(f'[GRASP V2] Policy device: {policy.torch_policy.device}. Learned positive position/pressure gains; '
                  'position/pressure feedback, grasp choice, phases, orientation/IK and contact/slip guards remain programmed.',flush=True)
        stages=stage_limits(args.stage_steps,env.step_dt)
        required_hold_steps=hold_steps(env.step_dt)
        if args.stage_steps < required_hold_steps:
            raise ValueError('--stage_steps must allow at least 2 simulated seconds for the hold check')
        print(f'[GRASP V2] Control dt={env.step_dt:g}s; maximum per phase={args.stage_steps} steps '
              f'({args.stage_steps*env.step_dt:g}s); required stable hold=2s',flush=True)
        print(f'[GRASP V2] Absolute IK trajectory: free={args.free_speed*1000:g} mm/s; '
              f'near peg={args.contact_speed*1000:g} mm/s; rotation={args.rotation_speed:g} rad/s. '
              'Speeds describe the reference; actual motion is measured. '
              'phase_error refers to the current waypoint; grasp_distance refers to the peg grasp site.',flush=True)
        print(f'[GRASP V2] Preload: {args.grip_force:g} N per finger for {args.preload_seconds:g}s; '
              f'lift={args.lift_speed*1000:g} mm/s, paused on weak pressure; '
              'abort on 1.5 mm slip, lost contact or excessive force.',flush=True)
        with trace_path.open('x') as trace,torch.inference_mode():
            for episode in range(args.episodes):
                env.reset()
                obj = env.scene['object'].data
                start_pos,start_quat = obj.root_pos_w.clone(),obj.root_quat_w.clone()
                local = torch.tensor([plan['tip_local_m']],device=env.device,dtype=start_pos.dtype)
                grasp = start_pos+quat_apply(start_quat,local)
                orient = quat_mul(start_quat,torch.tensor([plan['tool_quaternion_local_wxyz']],device=env.device))
                direction=torch.tensor([plan['approach_direction_local']],device=env.device,dtype=start_pos.dtype)
                outside=grasp+quat_apply(start_quat,.015*direction)
                above = outside.clone()
                above[:,2] = max(float(above[0,2])+.025,.055)
                lift = grasp.clone();lift[:,2] += .035
                angle = plan['open_angle_rad']
                grip=grip_controller(args.mode,angle,args.grip_force,args.max_force,
                                    args.friction,args.mass_g/1000,args.preload_seconds)
                initial_bottom = telemetry(env,profile,args.arm)['peg_bottom_m']
                reference = None;grasp_passed=False;failed=None;records=[];peak=np.zeros(2)
                for name,budget in stages:
                    stable=0;achieved=False
                    row=None
                    watchdog=MotionWatchdog()
                    initial_frame=env.scene[f'ee_{args.arm}_frame'].data
                    trajectory=PoseTrajectory(initial_frame.target_pos_w[0,0].cpu().numpy(),
                                              initial_frame.target_quat_w[0,0].cpu().numpy())
                    paused=False;grip_state=None
                    contact_motion=name in ('reach','close','lift','hold','lower','release')
                    target = above if name in ('above','retreat') else outside if name=='outside' else lift if name in ('lift','hold') else grasp
                    if name == 'settle':
                        target = env.scene[f'ee_{args.arm}_frame'].data.target_pos_w[:,0].clone()
                        target_rot = env.scene[f'ee_{args.arm}_frame'].data.target_quat_w[:,0].clone()
                    else:
                        target_rot = orient
                    if name=='close':
                        # Close at the measured arrival pose, without pushing
                        # the peg to erase the remaining waypoint error.
                        target=initial_frame.target_pos_w[:,0].clone()
                        target_rot=initial_frame.target_quat_w[:,0].clone()
                    for step in range(budget):
                        tick = time.monotonic()
                        if not app.is_running():
                            raise KeyboardInterrupt
                        row = telemetry(env,profile,args.arm)
                        previous_angle=angle
                        previous_reference=(np.asarray(row['tool_tip_w_m']).copy() if trajectory.paused
                                            else trajectory.position.copy())
                        policy_observation=(observation(row,name,target[0].cpu().numpy(),
                            target_rot[0].cpu().numpy(),previous_reference,previous_angle)
                            if policy or args.demonstrations else None)
                        forces = np.asarray(row['finger_normal_force_n'])
                        peak = np.maximum(peak,forces)
                        if np.max(forces) > args.max_force:
                            failed='excessive_finger_force';break
                        if name in ('close','lift','hold','lower'):
                            if policy:
                                grip.angle=previous_angle
                            grip_state=grip.update(row,name,env.step_dt)
                            angle=grip_state['jaw_target_rad']
                            if grip_state['reason']:
                                row.update(episode=episode,stage=name,step=step,
                                           grip_control=grip_state,jaw_target_rad=angle,
                                           sample_timing='pre-action abort; no further motion commanded')
                                trace.write(json.dumps(row,allow_nan=False)+'\n')
                                failed=grip_state['reason'];break
                            if name=='close' and grip_state.get('centering_position_w_m') is not None:
                                target=torch.as_tensor([grip_state['centering_position_w_m']],
                                                       device=env.device,dtype=target.dtype)
                        elif name in ('release','retreat'):
                            angle=min(plan['open_angle_rad'],angle+.0015)
                        pause_motion=name in ('lift','hold','lower') and not grip_state['motion_allowed']
                        linear_speed=(args.lift_speed if name in ('lift','hold') else
                                      args.contact_speed if contact_motion else args.free_speed)
                        waypoint_pos,waypoint_quat=trajectory.advance(
                            row['tool_tip_w_m'],row['tool_quaternion_wxyz'],
                            target[0].cpu().numpy(),target_rot[0].cpu().numpy(),env.step_dt,
                            linear_speed,
                            min(args.rotation_speed,.2) if contact_motion else args.rotation_speed,
                            contact=contact_motion,allow_motion=not pause_motion)
                        neural_action=None
                        if policy:
                            neural_action=policy.predict(policy_observation)
                            waypoint_pos,angle=motor_command(neural_action,
                                trajectory.position if pause_motion else previous_reference,
                                row['tool_tip_w_m'],previous_angle,env.step_dt,linear_speed,
                                contact_motion,not pause_motion)
                            trajectory.position=waypoint_pos.copy()
                        elif args.demonstrations:
                            demo_samples.append((episode,name,policy_observation,
                                teacher_action(previous_reference,waypoint_pos,previous_angle,angle,
                                               env.step_dt,linear_speed,not pause_motion)))
                        paused=pause_motion
                        command_pos=torch.as_tensor(waypoint_pos[None],device=env.device,dtype=target.dtype)
                        command_quat=torch.as_tensor(waypoint_quat[None],device=env.device,dtype=target_rot.dtype)
                        env.step(control(env,command_pos,command_quat,angle,args.arm))
                        row = telemetry(env,profile,args.arm)
                        pos_error,rot_error = compute_pose_error(
                            env.scene[f'ee_{args.arm}_frame'].data.target_pos_w[:,0],
                            env.scene[f'ee_{args.arm}_frame'].data.target_quat_w[:,0],target,target_rot,rot_error_type='axis_angle')
                        row.update(episode=episode,stage=name,step=step,jaw_target_rad=angle,
                                   neural_action=neural_action.tolist() if neural_action is not None else None,
                                   grip_control=grip_state,motion_paused=paused,
                                   stage_elapsed_s=(step+1)*env.step_dt,stage_budget_s=budget*env.step_dt,
                                   target_error_mm=float(pos_error.norm()*1000),orientation_error_rad=float(rot_error.norm()))
                        live_grasp=obj.root_pos_w+quat_apply(obj.root_quat_w,local)
                        row.update(phase_target_w_m=target[0].cpu().tolist(),
                                   command_position_w_m=waypoint_pos.tolist(),
                                   grasp_site_w_m=live_grasp[0].cpu().tolist(),
                                   grasp_site_distance_mm=float((env.scene[f'ee_{args.arm}_frame'].data.target_pos_w[:,0]-live_grasp).norm()*1000))
                        trace.write(json.dumps(row,allow_nan=False)+'\n')
                        if (step+1)%100==0:
                            trace.flush()
                        forces=np.asarray(row['finger_normal_force_n']);peak=np.maximum(peak,forces)
                        if forces.max()>args.max_force:
                            failed='excessive_finger_force';break
                        if row['peg_bottom_m']<.005:
                            failed='peg_dropped';break
                        if name=='settle':
                            achieved=step==budget-1
                        elif name in ('above','outside','reach','retreat','lower'):
                            reached=pose_reached(row['target_error_mm'],row['orientation_error_rad'])
                            stable=stable+1 if reached else 0
                            achieved=stable>=10
                        elif name=='close':
                            # The final sample must still show both loaded
                            # contacts; the dwell is measured by the controller.
                            vectors=np.asarray(row['finger_force_vectors_n'])
                            contact=bool((forces>=args.grip_force-.05).all()
                                and np.dot(*vectors)<-.25*np.prod(forces)
                                and valid_contact_pair(row,args.mode))
                            achieved=bool(grip_state['pressure_ready'] and contact)
                            if achieved:
                                reference=row
                                # Lift vertically from the actual loaded pose;
                                # do not correct a lateral IK error inside the pinch.
                                grasp=env.scene[f'ee_{args.arm}_frame'].data.target_pos_w[:,0].clone()
                                orient=env.scene[f'ee_{args.arm}_frame'].data.target_quat_w[:,0].clone()
                                lift=grasp.clone();lift[:,2]+=.035
                        elif name in ('lift','hold'):
                            evidence=bool(reference and physical_hold(row,initial_bottom,reference['peg_in_tool_m'])
                                          and valid_contact_pair(row,args.mode)
                                          and (forces>=args.grip_force-.05).all()
                                          and pose_reached(row['target_error_mm'],row['orientation_error_rad']))
                            if reference:
                                dot=abs(np.dot(row['peg_quat_in_tool'],reference['peg_quat_in_tool']))
                                evidence &= 2*np.arccos(np.clip(dot,-1,1))<.15
                            stable=stable+1 if evidence else 0
                            achieved=stable>=20 if name=='lift' else stable>=required_hold_steps
                            if name=='hold' and achieved:
                                grasp_passed=True
                        elif name=='release':
                            released=bool((forces<.01).all() and abs(row['peg_bottom_m']-.010)<.002 and row['peg_speed_m_s']<.02)
                            stable=stable+1 if released else 0
                            achieved=stable>=25 and angle>=plan['open_angle_rad']-.01
                        if step==0 or (step+1)%100==0:
                            print(f'[GRASP V2] {episode+1}/{args.episodes} {name}: '
                                  f'{step+1}/{budget} steps; phase_error={row["target_error_mm"]:.2f} mm; '
                                  f'grasp_distance={row["grasp_site_distance_mm"]:.2f} mm; '
                                  f'rotation_error={row["orientation_error_rad"]:.3f} rad; '
                                  f'force_N={[round(float(f),3) for f in forces]}; '
                                  f'paused={paused}; '
                                  f'slip_mm={grip_state["slip_mm"] if grip_state else 0.:.2f}; '
                                  f'closure_shift_mm={grip_state.get("closure_peg_displacement_mm",0.) if grip_state else 0.:.2f}; '
                                  f'centering_mm={grip_state.get("centering_offset_mm",0.) if grip_state else 0.:.2f}; '
                                  f'joint_limits={row["joints_near_limit"]}',flush=True)
                        if name in ('above','outside','reach','retreat','lower') and not achieved:
                            if watchdog.stalled((step+1)*env.step_dt,row['target_error_mm'],row['orientation_error_rad']):
                                failed=f'{name}_motion_stalled'
                                print(f'[GRASP V2] Motion stalled for 5 simulated seconds: phase={name}; '
                                      f'phase_error={row["target_error_mm"]:.2f} mm; '
                                      f'grasp_distance={row["grasp_site_distance_mm"]:.2f} mm; '
                                      f'rotation_error={row["orientation_error_rad"]:.3f} rad; '
                                      f'joints_near_limit={row["joints_near_limit"]}. '
                                      'Stopping; the remaining stage budget cannot fix a stationary pose.',flush=True)
                                break
                        if args.realtime:
                            time.sleep(max(0.,env.step_dt-(time.monotonic()-tick)))
                        if achieved:
                            break
                    records.append(dict(stage=name,steps=step+1,achieved=achieved,budget_steps=budget,
                        elapsed_s=(step+1)*env.step_dt,budget_s=budget*env.step_dt,
                        target_error_mm=row.get('target_error_mm') if row else None,
                        orientation_error_rad=row.get('orientation_error_rad') if row else None,
                        grip_control=grip_state,
                        grasp_site_distance_mm=row.get('grasp_site_distance_mm') if row else None,
                        joints_near_limit=row.get('joints_near_limit') if row else None))
                    print(f'[GRASP V2] {episode+1}/{args.episodes} {name}: achieved={achieved}; '
                          f'reason={failed or ("achieved" if achieved else "budget_exhausted")}; '
                          f'force_peak_N={peak.tolist()}',flush=True)
                    if failed or not achieved:
                        failed=failed or f'{name}_budget_exhausted';break
                    if name=='settle':
                        # Plan from the settled body, so reset fall does not change insertion depth.
                        start_pos=obj.root_pos_w.clone();start_quat=obj.root_quat_w.clone()
                        grasp=start_pos+quat_apply(start_quat,local)
                        orient=quat_mul(start_quat,torch.tensor([plan['tool_quaternion_local_wxyz']],device=env.device))
                        outside=grasp+quat_apply(start_quat,.015*direction)
                        above=outside.clone();above[:,2]=max(float(grasp[0,2])+.025,.055)
                        lift=grasp.clone();lift[:,2]+=.035
                        initial_bottom=row['peg_bottom_m']
                cases.append(dict(episode=episode,physical_grasp_hold_success=grasp_passed,
                    complete=failed is None,reason=failed,stages=records,force_peak_n=peak.tolist(),
                    terminal=telemetry(env,profile,args.arm)))
        if args.demonstrations:
            demo_result=save_demonstrations(args.demonstrations,demo_samples,cases,experiment)
            print(f'[GRASP V2] Demonstrations: {args.demonstrations}; '
                  f'complete successes={len(demo_result["episodes"])}; transitions={demo_result["transitions"]}',flush=True)
    except KeyboardInterrupt:
        interrupted=True
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        import traceback
        traceback.print_exc()
    finally:
        sources=sorted(Path(__file__).parent.glob('*.py'))+[PROFILE,MODEL]
        report=dict(schema='peg-contact-probe-2',strategy=(
            'learned positive xyz position/jaw pressure gains with programmed position/pressure feedback/phases/orientation/IK/contact guards'
            if policy else 'bounded absolute-pose IK trajectory and measured-contact controller; no learned policy'),
            learned_active_translation_and_jaw=policy is not None,
            programmed_goal_error_feedback=policy is not None,
            programmed_pressure_feedback=policy is not None,
            inference_device=str(policy.torch_policy.device) if policy else None,
            camera=dict(eye=list(args.camera_eye),target=list(args.camera_target)),
            policy_checkpoint=str(args.policy.resolve()) if args.policy else None,
            policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest() if args.policy else None,
            expert_xyz_jaw_actions_applied=policy is None,
            programmed_grasp_selection_phases_orientation=True,demonstrations=demo_result,
            code_sha256=hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest(),
            geometry_sha256=hashlib.sha256(PROFILE.read_bytes()).hexdigest(),peg_asset_sha256=profile['asset_sha256'],
            robot_asset_sha256=profile['robot']['asset_sha256'] if profile['robot'] else None,
            physics_model='rigid mesh-derived compound body with compliant normal contacts; no FEM deformation',
            actual_physx_materials=getattr(env,'grasp_material_evidence',None),
            initial_physx_materials=getattr(env,'grasp_material_initial_evidence',None),
            assisted_grasp=False,object_pose_servo=False,fixed_or_magnetic_attachment=False,
            source='L5',candidate=plan,arrival_geometry_check=arrival,
            kinematic_route_check=reachability,kinematics_sha256=hashlib.sha256(MODEL.read_bytes()).hexdigest(),
            parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            versions={name:importlib.metadata.version(name) for name in ('isaacsim','isaaclab','torch')},
            force_evidence='filtered finger-to-peg normal contact forces, sampled at control rate (100 Hz)',
            cases=cases,requested_episodes=args.episodes,episodes=len(cases),interrupted=interrupted,error=error,
            complete=len(cases)==args.episodes and not interrupted and error is None,
            physical_grasp_hold_success=sum(c['physical_grasp_hold_success'] for c in cases),trace=str(trace_path))
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(f'[GRASP V2] Report: {args.output}',flush=True)
        try:
            if env is not None:
                env.close()
        finally:
            app.close()
    return 1 if error or interrupted or not cases or any(not c['complete'] for c in cases) else 0


if __name__=='__main__':
    raise SystemExit(main())
