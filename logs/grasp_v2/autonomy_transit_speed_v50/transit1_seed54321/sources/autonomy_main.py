"""Collect native demonstrations/DAgger corrections and evaluate free BC motors."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import tempfile
import numpy as np

REPO=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')

def record_allocated_result(completed,episode,result,requested):
    if not 0<=episode<requested:return False
    if episode in completed:raise ValueError('Allocated episode completed twice')
    completed[episode]=result
    return True

def learner_execution_mask(ages,stages,fraction,until_stage,prefix_steps=0,draws=None):
    """Collection only: a continuous rollout, then exclusive expert recovery."""
    if prefix_steps:return ages<prefix_steps
    if draws is None:
        import torch
        draws=torch.rand(len(ages),device=ages.device)
    return (draws<fraction)&(stages<until_stage)

def save_completed_demonstrations(out,chunks,completed,conditions,manifest,schema,args,partial=False):
    """Persist complete successful trajectories; never train on unfinished ones."""
    import torch
    successful=[k for k,v in completed.items() if v['success']]
    if not successful:return None
    joined={k:torch.cat([c[k] for c in chunks]).cpu().numpy() for k in chunks[0]}
    keep=np.isin(joined['episode'],successful)
    data={k:v[keep] for k,v in joined.items()}
    validation_episodes=successful[4::5] or successful[-1:]
    data['validation']=np.isin(data['episode'],validation_episodes)
    metadata=dict(schema=schema,skill=args.skill,assisted_grasp=bool(getattr(args,'grasp_support',0.)),teacher=True,learner_fraction=args.learner_fraction,
        collection_complete=not partial,episodes=successful,validation_episodes=validation_episodes,transitions=int(keep.sum()),
        parameters=manifest['parameters'],resets={str(k):conditions[k] for k in successful},source_hashes=manifest['source_hashes'])
    filename='completed_demonstrations.npz' if partial else 'demonstrations.npz'
    temporary=out/(filename+'.tmp')
    with temporary.open('wb') as stream:np.savez_compressed(stream,**data,metadata=json.dumps(metadata))
    temporary.replace(out/filename)
    return metadata

def snapshot(out):
    folder=out/'sources';folder.mkdir(parents=True,exist_ok=True)
    names=['rl_core.py','rl_env.py','rl_runtime.py','geometry.py','peg_profile.json','replay_full.py','replay.py']
    for path in list(HERE.glob('autonomy_*.py'))+[HERE/name for name in names]:shutil.copy2(path,folder/path.name)
    return {p.name:sha(p) for p in folder.iterdir() if p.is_file()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('collect','train','fit_jaws','fit_router','fit_contact','evaluate','play','repackage'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--policy',type=Path)
    p.add_argument('--datasets',type=Path,nargs='+');p.add_argument('--num_envs',type=int,default=16)
    p.add_argument('--episodes',type=int,default=32);p.add_argument('--episode_steps',type=int,default=5000)
    p.add_argument('--jitter_mm',type=float,default=.5);p.add_argument('--yaw_deg',type=float,default=2.)
    p.add_argument('--sources',nargs='+',default=['L5']);p.add_argument('--targets',nargs='+',default=['R2'])
    p.add_argument('--learner_fraction',type=float,default=0.,help='DAgger probability of executing the learner per environment/step')
    p.add_argument('--learner_until_stage',type=int,default=19,help='Limit DAgger perturbations to approach stages while collecting initial recoveries')
    p.add_argument('--learner_prefix_steps',type=int,default=0,help='Collection-only alternative: execute a continuous learner prefix, then let the expert recover without alternating controllers')
    p.add_argument('--sync_learner_reference',action='store_true',help='Collection-only: reanchor expert private motor references during the learner prefix and at handoff')
    p.add_argument('--receiver_recovery_handoff',action='store_true',help='Collection-only: start expert receiver reach from a measured stable donor lift at the end of the learner prefix')
    p.add_argument('--updates',type=int,default=6000);p.add_argument('--batch_size',type=int,default=1024)
    p.add_argument('--architecture',choices=('mlp','neighbors','gru','pose','mixture','linear_options'),default='mlp')
    p.add_argument('--future_steps',type=int,default=0,help='BC target from future measured poses; no future observations at inference')
    p.add_argument('--phase_endpoint',action='store_true',help='For pose BC, label measured stage endpoints offline; stage is NOT an actor input')
    p.add_argument('--pose_entry_supervision',action='store_true',help='BC only: fit latched option poses from measured entry states; gate and jaws retain current-state inputs')
    p.add_argument('--exclude_joint_shortcuts',action='store_true',help='BC option gate: use measured geometry/contact; exclude joint and velocity correlations')
    p.add_argument('--skill',choices=('lift','handover','full'),default='lift')
    p.add_argument('--balance_direction',type=float,choices=(-1.,1.),default=1.,help='Collection-only receiver contact centering experiment')
    p.add_argument('--receiver_mode',choices=('wall','vertex'),default='wall',help='Data-collection receiver grasp candidate; inference decisions stay in the actor')
    p.add_argument('--receiver_wall_angle',type=float,choices=(60.,300.),default=60.,help='Collection-only upper/lower right wall grasp candidate in degrees')
    p.add_argument('--receiver_wall_depth',type=float,default=1.,help='Collection-only wall insertion depth below the upper rim in mm')
    p.add_argument('--receiver_wall_tilt',type=float,default=20.,help='Collection-only wall tool entry tilt in degrees')
    p.add_argument('--receiver_open_angle',type=float,help='Collection-only wall jaw opening in radians')
    p.add_argument('--receiver_release_bottom',type=float,default=.010,help='Collection-only insertion bottom height in metres before release; success still requires free stable seating at 0.010 m')
    p.add_argument('--receiver_pressure_min',type=float,default=.45,help='Collection-only lower receiver pressure in N')
    p.add_argument('--receiver_pressure_max',type=float,default=.55,help='Collection-only upper receiver pressure in N')
    p.add_argument('--donor_release_rate',type=float,default=.02,help='Collection-only gradual donor release rate in rad/s')
    p.add_argument('--fixed_donor_during_contact',action='store_true',help='Collection-only experiment: preserve donor tool pose during receiver approach and closure')
    p.add_argument('--recapture_receiver_after_transfer',action='store_true',help='Collection-only: measure the loaded receiver-to-peg transform after the donor retreats')
    p.add_argument('--implementation_change',default='Preserve XYZ/rotation direction when bounding long pose commands',help='Description of the current-code fork when repackaging unchanged weights')
    p.add_argument('--transition_data',type=Path,nargs='+',help='Repackage: infer option-transition memory from complete training demonstrations')
    p.add_argument('--stationary_data',type=Path,nargs='+',help='Repackage: infer which tool references remain stationary from whole training demonstrations')
    p.add_argument('--stationary_position_tolerance_mm',type=float,default=.5,help='Training reference excursions below this limit share the previous neural goal')
    p.add_argument('--stationary_orientation_tolerance_rad',type=float,default=.02,help='Training rotation excursions below this limit share the previous neural orientation')
    p.add_argument('--pickup_policy',type=Path,help='Repackage: embed a preserved six-option neural pickup decoder in a full actor')
    p.add_argument('--use_pickup_gate',action='store_true',help='Repackage: retain the embedded pickup neural decisions until the full learned router selects continuation')
    p.add_argument('--track_stationary_adjustments',action='store_true',help='Repackage: retain the entry anchor and follow bounded changes in learned contact pose predictions')
    p.add_argument('--relative_pose_data',type=Path,nargs='+',help='Repackage: fit a peg-frame BC goal decoder from whole native training visits')
    p.add_argument('--relative_pose_options',type=int,nargs='+',default=[7,8,9,10,11],help='Options whose receiver poses are fitted in the observed peg frame')
    p.add_argument('--router_jaw_inputs',action='store_true',help='Router fitting: include measured jaw positions/references to distinguish donor release and retreat')
    p.add_argument('--target_object_data',type=Path,nargs='+',help='Repackage: fit object destinations from complete training endpoints; decode through measured receiver grasp')
    p.add_argument('--target_object_options',type=int,nargs='+',default=[14,15],help='Options whose object destinations are fitted relative to the target')
    p.add_argument('--track_relative_pose_changes',action='store_true',help='Repackage: follow bounded changes of fitted peg-relative goals after entry')
    p.add_argument('--diagnostic_events',action='store_true',help='Log all eligible environments at learned option changes and every 50 steps; no actor input changes')
    p.add_argument('--diagnostic_interval',type=int,default=50,help='Diagnostic-only physical sampling interval in control steps')
    p.add_argument('--grasp_support',type=float,default=0.,help='Explicit bounded grasp spring strength 0..1; all nonzero runs are assisted')
    p.add_argument('--support_profile',choices=('gentle','firm'),default='gentle',help='Assisted grasp force/stiffness preset; strength zero remains native')
    p.add_argument('--placement_support',action='store_true',help='Explicit temporary centering guide after free release near the target; assisted only')
    p.add_argument('--free_speed_scale',type=float,default=1.,help='Reference translation/rotation multiplier away from contact; original checkpoint uses 1')
    p.add_argument('--contact_speed_scale',type=float,default=1.,help='Reference translation/rotation multiplier in contact; jaws and hold times stay unchanged')
    p.add_argument('--motion_profile',choices=('contact','clearance','travel'),default='contact',help='Experimental transit caps; travel preserves original orientation and approach minima')
    p.add_argument('--transport_speed_scale',type=float,default=1.,help='Clearance profile only: stable isolated grasp transport multiplier, 1..4; interaction stays slow')
    p.add_argument('--latch_option_poses',action='store_true',help='Repackage: choose and retain a learned pose on each learned option change; jaws remain closed-loop learned outputs')
    p.add_argument('--pressure_supervision',action='store_true',help='Fit jaws: imitate offline expert pressure decisions with categorical rates and force calibration')
    p.add_argument('--pressure_min',type=float,default=.30,help='Training-only lower pressure label boundary in N')
    p.add_argument('--pressure_max',type=float,default=.40,help='Training-only upper pressure label boundary in N')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda:0')
    p.add_argument('--headless',action='store_true');p.add_argument('--realtime',action='store_true');p.add_argument('--video',action='store_true')
    args=p.parse_args()
    if not np.isfinite(args.grasp_support) or not 0<=args.grasp_support<=1:p.error('Grasp support must be finite and within 0..1')
    if args.placement_support and not args.grasp_support:p.error('Placement guide requires nonzero assistance')
    free_max=4 if args.motion_profile!='contact' else 2
    if not np.isfinite(args.free_speed_scale) or not 1<=args.free_speed_scale<=free_max:
        p.error(f'Free speed scale must be finite and within 1..{free_max}')
    if not np.isfinite(args.contact_speed_scale) or not 1<=args.contact_speed_scale<=2:
        p.error('Contact speed scale must be finite and within 1..2')
    if not np.isfinite(args.transport_speed_scale) or not 1<=args.transport_speed_scale<=4:
        p.error('Transport speed scale must be finite and within 1..4')
    if args.motion_profile!='contact' and args.contact_speed_scale!=1:
        p.error('Transit profiles preserve the original slow contact speed')
    if args.motion_profile=='contact' and args.transport_speed_scale!=1:
        p.error('Transport speed scale requires the clearance profile')
    if args.diagnostic_interval<1:p.error('Diagnostic interval must be positive')
    if not args.device.startswith('cuda'):p.error('CUDA required')
    if min(args.num_envs,args.episodes,args.episode_steps,args.updates,args.batch_size)<1:p.error('Positive counts required')
    if not 0<=args.learner_fraction<=1:p.error('learner_fraction must be 0..1')
    if not 0<=args.learner_prefix_steps<args.episode_steps:p.error('Learner prefix must be shorter than the episode budget')
    if args.learner_prefix_steps and (args.mode!='collect' or not args.policy):p.error('Learner prefix requires collection with a policy')
    if args.sync_learner_reference and (args.mode!='collect' or not args.policy):p.error('Reference synchronization requires collection with a policy')
    if args.receiver_recovery_handoff and (args.mode!='collect' or args.skill!='full' or not args.learner_prefix_steps or not args.sync_learner_reference):
        p.error('Receiver recovery requires full collection, a synchronized learner prefix and a policy')
    if not 0<=args.future_steps<=100:p.error('future_steps must be 0..100')
    if args.phase_endpoint and args.architecture!='pose':p.error('phase_endpoint requires architecture pose')
    if args.pose_entry_supervision and (args.architecture not in ('linear_options','mixture') or not args.latch_option_poses):p.error('Entry supervision requires option architecture and latched poses')
    if args.exclude_joint_shortcuts and args.architecture not in ('linear_options','mixture'):p.error('Joint exclusion requires option architecture')
    if args.jitter_mm<0 or args.jitter_mm>1 or not 0<=args.yaw_deg<=5:p.error('First curriculum: jitter 0..1 mm and yaw 0..5 deg')
    if (args.mode in ('evaluate','play') or args.learner_fraction>0) and not args.policy:p.error('Specify --policy')
    if args.video and (args.num_envs!=1 or args.episodes!=1):p.error('Video needs one environment and episode')
    if not .1<=args.pressure_min<args.pressure_max<=.8:p.error('Pressure training band must satisfy 0.1 <= min < max <= 0.8 N')
    if not .1<=args.receiver_pressure_min<args.receiver_pressure_max<=.8:p.error('Invalid receiver pressure band')
    if not 0<args.donor_release_rate<=.15:p.error('Donor release rate must be 0..0.15 rad/s')
    if not .5<=args.receiver_wall_depth<=8. or not 0<=args.receiver_wall_tilt<=30.:p.error('Invalid receiver wall depth or tilt')
    if args.receiver_open_angle is not None and not 0<args.receiver_open_angle<=.5:p.error('Invalid receiver jaw opening')
    if not .010<=args.receiver_release_bottom<=.035:p.error('Insertion bottom before release must be 0.010..0.035 m')
    if not 0<args.stationary_position_tolerance_mm<=2. or not 0<args.stationary_orientation_tolerance_rad<=.15:
        p.error('Stationary reference tolerances must be 0..2 mm and 0..0.15 rad')
    if args.mode=='repackage':
        if not args.policy:p.error('Specify --policy')
        parent=json.loads((args.policy.parent/'manifest.json').read_text())
        if sha(args.policy)!=parent['policy_sha256']:raise ValueError('Parent weights changed')
        for name,digest in parent['source_hashes'].items():
            if sha(args.policy.parent/'sources'/name)!=digest:raise ValueError('Parent source changed')
        if sha(REPO/'milestones/checkpoint3-full-assisted.json')!=parent['physics_manifest_sha256']:
            raise ValueError('Preserved physics changed')
        from autonomy_learning import load_policy
        import torch
        net=load_policy(args.policy,args.device)
        transition_details={}
        if args.stationary_data:
            from autonomy_learning import demonstrated_stationary_goals
            if not hasattr(net,'preserve_stationary_goals'):raise ValueError('Stationary memory requires option actor')
            position,orientation,evidence=demonstrated_stationary_goals(args.stationary_data,net.options,args.device,
                args.stationary_position_tolerance_mm*.001,args.stationary_orientation_tolerance_rad)
            net.stationary_position.copy_(position);net.stationary_orientation.copy_(orientation);net.preserve_stationary_goals.fill_(True)
            transition_details.update(stationary_reference_evidence=evidence,
                stationary_dataset_hashes={str(path):sha(path) for path in args.stationary_data},
                stationary_position_tolerance_m=args.stationary_position_tolerance_mm*.001,
                stationary_orientation_tolerance_rad=args.stationary_orientation_tolerance_rad,
                stationary_reference_memory_learned_from_training_only=True)
        if args.transition_data:
            from autonomy_learning import demonstrated_transitions
            if not hasattr(net,'allowed_transitions'):raise ValueError('Transition memory requires option actor')
            allowed,initial,counts=demonstrated_transitions(args.transition_data,net.options,args.device)
            net.allowed_transitions.copy_(allowed);net.initial_option.fill_(initial)
            transition_details.update(transition_counts=counts.cpu().tolist(),initial_option=initial,
                transition_dataset_hashes={str(path):sha(path) for path in args.transition_data})
        if args.pickup_policy:
            from autonomy_learning import OptionPosePolicy
            pickup_manifest=json.loads((args.pickup_policy.parent/'manifest.json').read_text())
            if sha(args.pickup_policy)!=pickup_manifest['policy_sha256']:raise ValueError('Pickup weights changed')
            for name,digest in pickup_manifest['source_hashes'].items():
                if sha(args.pickup_policy.parent/'sources'/name)!=digest:raise ValueError('Pickup source changed')
            if pickup_manifest['physics_manifest_sha256']!=parent['physics_manifest_sha256']:raise ValueError('Pickup physics incompatible')
            pickup=load_policy(args.pickup_policy,args.device)
            if not isinstance(net,OptionPosePolicy) or net.options!=19 or not net.linear_goals or net.jaw_rate_head is None:
                raise ValueError('Composition requires full linear option actor with learned jaws')
            if not isinstance(pickup,OptionPosePolicy) or pickup.options!=6 or not pickup.linear_goals or pickup.jaw_rate_head is None or pickup.pickup_decoder is not None:
                raise ValueError('Composition requires six-option pickup actor with learned jaws')
            net.pickup_decoder=pickup.requires_grad_(False)
            # Keep the original pickup pose decoder behavior; stationary
            # memory continues only for the remaining full-task options.
            net.stationary_position[:6]=False;net.stationary_orientation[:6]=False
            transition_details.update(pickup_parent_policy=str(args.pickup_policy.resolve()),
                pickup_parent_sha256=sha(args.pickup_policy),pickup_neural_parameters_frozen=True,
                composition='Learned full router; embedded pickup pose/jaw heads for options 0..5; full pose/jaw heads for 6..18')
        if args.use_pickup_gate:
            if getattr(net,'pickup_decoder',None) is None or net.option_router is None:raise ValueError('Pickup gate requires embedded pickup and full learned router')
            net.use_pickup_gate.fill_(True)
            transition_details.update(pickup_neural_gate_retained=True,
                composition='Embedded pickup neural gate/pose/jaws; learned full router selects continuation from its final option; full motor heads for remaining options')
        if getattr(net,'pickup_decoder',None) is not None:
            net.stationary_position[:6]=False;net.stationary_orientation[:6]=False
        if args.track_stationary_adjustments:
            if not args.stationary_data or not bool(net.latch_option_poses) or not bool(net.preserve_stationary_goals):
                raise ValueError('Adjustment memory needs stationary training data and latched neural goals')
            fine_position,fine_orientation,_=demonstrated_stationary_goals(args.stationary_data,net.options,args.device)
            net.adjust_stationary_position.copy_(net.stationary_position&~fine_position)
            net.adjust_stationary_orientation.copy_(net.stationary_orientation&~fine_orientation)
            net.track_stationary_adjustments.fill_(True)
            transition_details.update(neural_pose_adjustment_memory=True,neural_pose_adjustment_bounds_m=.002,
                neural_pose_adjustment_rotation_bound_rad=.13,
                adjust_stationary_position=net.adjust_stationary_position.cpu().tolist(),
                adjust_stationary_orientation=net.adjust_stationary_orientation.cpu().tolist())
        if args.relative_pose_data:
            from autonomy_learning import demonstrated_relative_poses
            if net.options!=19 or any(option<6 or option>=19 for option in args.relative_pose_options):raise ValueError('Relative full decoder requires compatible full options')
            poses,mask,evidence=demonstrated_relative_poses(args.relative_pose_data,net.options,1,args.relative_pose_options,args.device)
            with torch.no_grad():net.relative_pose_goals.copy_(poses);net.relative_pose_mask.copy_(mask)
            transition_details.update(relative_pose_decoder='Peg-frame transforms fitted by supervised pose averaging, whole training episodes only',
                relative_pose_evidence=evidence,relative_pose_dataset_hashes={str(path):sha(path) for path in args.relative_pose_data},
                original_neural_parameters_unchanged=True,new_relative_decoder_parameters_fitted=True)
        if args.target_object_data:
            from autonomy_learning import demonstrated_target_objects
            if net.options!=19 or any(option<6 or option>=19 for option in args.target_object_options):raise ValueError('Object destination decoder requires compatible full options')
            goals,mask,evidence=demonstrated_target_objects(args.target_object_data,net.options,args.target_object_options,args.device)
            with torch.no_grad():net.target_object_goals.copy_(goals);net.target_object_mask.copy_(mask)
            transition_details.update(target_object_decoder='Object destinations fitted relative to target; tool goals decoded through measured tool-to-peg pose; tool motors only',
                target_object_evidence=evidence,target_object_dataset_hashes={str(path):sha(path) for path in args.target_object_data},
                original_neural_parameters_unchanged=True,new_target_decoder_parameters_fitted=True)
        if args.track_relative_pose_changes:
            if not bool(net.relative_pose_mask.any()) or not bool(net.latch_option_poses):
                raise ValueError('Relative tracking requires fitted relative poses and latched goals')
            net.track_stationary_adjustments.fill_(True)
            net.adjust_stationary_position.logical_or_(net.relative_pose_mask)
            net.adjust_stationary_orientation.logical_or_(net.relative_pose_mask)
            transition_details.update(fitted_relative_pose_change_tracking=True,
                relative_tracking_position_bound_m=.002,relative_tracking_rotation_bound_rad=.13,
                relative_tracking_mask=net.relative_pose_mask.cpu().tolist())
        if args.latch_option_poses:
            if not hasattr(net,'latch_option_poses'):raise ValueError('Pose memory requires option actor')
            net.latch_option_poses.fill_(True)
            transition_details['latch_option_poses']=True
        with torch.inference_mode():
            action=net(net.mean[None])
            if action.shape!=(1,14) or not bool(torch.isfinite(action).all()):raise ValueError('Invalid actor')
        args.output.mkdir(parents=True,exist_ok=False)
        if args.transition_data or args.latch_option_poses or args.stationary_data or args.pickup_policy or args.use_pickup_gate or args.track_stationary_adjustments or args.relative_pose_data or args.target_object_data or args.track_relative_pose_changes:
            saved=torch.load(args.policy,map_location=args.device,weights_only=True)
            saved['state_dict']=net.state_dict();torch.save(saved,args.output/'policy.pt')
        else:shutil.copy2(args.policy,args.output/'policy.pt')
        cfg=json.loads((args.policy.parent/'training.json').read_text())
        fitted_decoder=bool(args.relative_pose_data or args.target_object_data)
        cfg.update(retrained=fitted_decoder,neural_parameters_unchanged=not fitted_decoder,implementation_change=args.implementation_change,**transition_details,
                   parent_policy=str(args.policy.resolve()),physical_evaluation_required=True)
        write(args.output/'training.json',cfg)
        parent.update(source_hashes=snapshot(args.output),parent_checkpoint=str(args.policy.resolve()),policy_sha256=sha(args.output/'policy.pt'),
            parent_policy_sha256=sha(args.policy),**transition_details,
            implementation_change=cfg['implementation_change'],retrained=cfg['retrained'])
        write(args.output/'manifest.json',parent)
        return 0
    if args.mode in ('train','fit_jaws','fit_router','fit_contact'):
        if not args.datasets:p.error('Specify datasets')
        parent=None
        if args.mode in ('fit_jaws','fit_router','fit_contact'):
            if not args.policy:p.error('Specify --policy')
            parent=json.loads((args.policy.parent/'manifest.json').read_text())
            if sha(args.policy)!=parent['policy_sha256']:raise ValueError('Parent weights changed')
            for name,digest in parent['source_hashes'].items():
                if sha(args.policy.parent/'sources'/name)!=digest:raise ValueError('Parent source changed')
            if sha(REPO/'milestones/checkpoint3-full-assisted.json')!=parent['physics_manifest_sha256']:raise ValueError('Preserved physics changed')
        for dataset in args.datasets:
            with np.load(dataset,allow_pickle=False) as archive:
                if json.loads(str(archive['metadata']))['skill']!=args.skill:
                    p.error('Dataset skill must match --skill')
        # Execute the snapshot, not files that might be edited during a long
        # training run. The preserved source must match the experiment code.
        with tempfile.TemporaryDirectory(prefix='peg_autonomy_train_',dir='/tmp') as frozen:
            hashes=snapshot(Path(frozen));folder=Path(frozen)/'sources'
            sys.path.insert(0,str(folder))
            from autonomy_learning import train,fit_jaw_rate,fit_option_router,SCHEMA
            if args.mode=='fit_router':fit_option_router(args.policy,args.datasets,args.output,args.device,args.updates,args.batch_size,args.seed,args.router_jaw_inputs)
            elif args.mode=='fit_jaws':fit_jaw_rate(args.policy,args.datasets,args.output,args.device,args.updates,args.batch_size,args.seed,args.pressure_supervision,args.pressure_min,args.pressure_max)
            elif args.mode=='fit_contact':
                from autonomy_learning import fit_contact_pose
                fit_contact_pose(args.policy,args.datasets,args.output,args.device,args.updates,args.batch_size,args.seed)
            else:train(args.datasets,args.output,args.device,args.updates,args.batch_size,args.seed,args.architecture,args.future_steps,args.phase_endpoint,args.latch_option_poses,args.pose_entry_supervision,args.exclude_joint_shortcuts)
            shutil.copytree(folder,args.output/'sources')
            write(args.output/'manifest.json',dict(schema=SCHEMA,source_hashes=hashes,policy_sha256=sha(args.output/'policy.pt'),
                dataset_hashes={str(path):sha(path) for path in args.datasets},
                physics_manifest_sha256=sha(REPO/'milestones/checkpoint3-full-assisted.json'),
                skill=args.skill,assisted_grasp=False,programmed_task_controller=False,
                parent_checkpoint=str(args.policy) if parent else None,parent_policy_sha256=parent['policy_sha256'] if parent else None))
        return 0
    from replay_full import verify
    base,bundle=verify(REPO/'milestones/checkpoint3-full-assisted.json')
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_hashes=snapshot(args.output),physics_manifest_sha256=sha(REPO/'milestones/checkpoint3-full-assisted.json'),
        assisted_grasp=bool(args.grasp_support),programmed_task_controller=args.mode=='collect',skill=args.skill,
        policy_sha256=sha(args.policy) if args.policy else None)
    write(args.output/'manifest.json',manifest)
    sys.path.insert(0,str(args.output/'sources'))
    scripts=bundle/'workflows/robotic_surgery/scripts';ext=scripts/'simulation/exts'
    sys.path[:0]=[str(scripts),str(ext/'robotic.surgery.tasks'),str(ext/'robotic.surgery.assets')]
    from isaaclab.app import AppLauncher
    app=AppLauncher(argparse.Namespace(headless=args.headless or args.video,enable_cameras=args.video,device=args.device)).app
    env=None;vec=None
    try:
        import torch
        torch.set_num_threads(8);torch.manual_seed(args.seed)
        from rl_runtime import make_env,initialize_contact_physics
        from rl_core import observe
        from rl_core import apply
        from geometry import contact_candidate
        from autonomy_env import AutonomyEnv
        from autonomy_teacher import PickupTeacher
        from autonomy_learning import load_policy,physical_observation,SCHEMA
        physical=argparse.Namespace(**base['parameters']);physical.num_envs=args.num_envs;physical.seed=args.seed;physical.device=args.device
        physical.robot_usd=bundle/'data/Isaac/Healthcare/0.5.0/132c82d/Robots/dVRK/PSM/psm.usd'
        profile=json.loads((bundle/'tools/peg_grasp_v2/peg_profile.json').read_text())
        env=make_env(physical,profile);initialize_contact_physics(env,physical.friction)
        env.sim.set_camera_view(eye=(.025,.32,.25),target=(.025,0.,.035))
        if args.video:
            from replay import EpisodeVideo
            EpisodeVideo(env,args.output/'inference_front.mp4')
        vec=AutonomyEnv(env,profile,args.skill,args.episode_steps,args.seed,args.jitter_mm,args.yaw_deg,args.sources,args.targets,
            args.free_speed_scale,args.contact_speed_scale,args.motion_profile,args.transport_speed_scale)
        print('[AUTONOMY] Motion reference: '+json.dumps(vec.reference.settings()),flush=True)
        if args.grasp_support:
            from autonomy_support import GraspSpring
            env.profile=profile
            vec.support=GraspSpring(env,args.grasp_support,args.support_profile,args.placement_support)
            vec.progress.grasp_support=vec.support
        vec.balance_direction=args.balance_direction
        vec.receiver_mode=args.receiver_mode
        vec.receiver_wall_angle=args.receiver_wall_angle
        vec.receiver_wall_depth=args.receiver_wall_depth;vec.receiver_wall_tilt=args.receiver_wall_tilt
        vec.receiver_open_angle=args.receiver_open_angle;vec.receiver_release_bottom=args.receiver_release_bottom
        vec.receiver_pressure_min=args.receiver_pressure_min;vec.receiver_pressure_max=args.receiver_pressure_max
        vec.donor_release_rate=args.donor_release_rate
        vec.fixed_donor_during_contact=args.fixed_donor_during_contact
        vec.recapture_receiver_after_transfer=args.recapture_receiver_after_transfer
        if args.skill!='lift':
            from autonomy_full_teacher import FullTeacher
            teacher_type=FullTeacher
        else:teacher_type=PickupTeacher
        obs=vec.reset();teacher=teacher_type(vec) if args.mode=='collect' else None
        net=load_policy(args.policy,args.device) if args.policy else None
        chunks=[];episode_id=torch.arange(args.num_envs,device=args.device);next_id=args.num_envs
        completed={};conditions={int(i):dict(vec.reset_records[int(i)]) for i in range(args.num_envs)};step=0
        saved_successes=0
        logged_options=None
        previous_learner=torch.zeros(args.num_envs,device=args.device,dtype=torch.bool)
        probe=[];local=torch.tensor(contact_candidate(profile,'wall',2,.5,1)['tip_local_m'],device=args.device,dtype=torch.float32)
        rollout_started=time.monotonic()
        with torch.inference_mode():
            while len(completed)<args.episodes:
                tick=time.monotonic();x=physical_observation(torch.as_tensor(obs,device=args.device))
                if teacher:
                    use=None
                    if net is not None:
                        use=learner_execution_mask(vec.age,teacher.stage,args.learner_fraction,args.learner_until_stage,args.learner_prefix_steps)
                        if args.sync_learner_reference:
                            teacher.sync_executed_reference(vec.reference,use|previous_learner)
                        if args.receiver_recovery_handoff:
                            teacher.begin_receiver_recovery(vec.reference,previous_learner&~use)
                    label=teacher.action();stage=teacher.stage.clone();executed=label
                    if net is not None:
                        executed=torch.where(use[:,None],net(x),label)
                        previous_learner=use.clone()
                    chunks.append(dict(obs=x.clone(),actions=label.clone(),stage=stage,episode=episode_id.clone()))
                else:executed=net(x)
                # Allocate a fixed set of episode IDs. Fast environments must
                # not replace a slow failing episode with repeated successes.
                eligible=episode_id<args.episodes
                executed=torch.where(eligible[:,None],executed,torch.zeros_like(executed))
                if args.diagnostic_events and net is not None:
                    choice=getattr(net,'selected_option',None)
                    if choice is not None:
                        changed=logged_options is None or bool(((choice!=logged_options)&eligible).any())
                        if changed or step%args.diagnostic_interval==0:
                            ids=eligible.nonzero().flatten()
                            event=dict(diagnostic_only=True,step=step,episode_ids=episode_id[ids].cpu().tolist(),
                                options=choice[ids].cpu().tolist(),physical_observations=x[ids].cpu().tolist(),
                                actions=executed[ids].cpu().tolist(),learned_goals=net.current_goal[ids].cpu().tolist())
                            with (args.output/'policy_events.jsonl').open('a') as stream:
                                stream.write(json.dumps(event)+'\n')
                        logged_options=choice.clone()
                if step%50==0:
                    s=vec.state;site=s['peg']+apply(s['peg_q'],local.expand(args.num_envs,-1))
                    probe.append(dict(step=step,episode=int(episode_id[0]),tip_m=s['tip'][0].cpu().tolist(),
                        peg_m=s['peg'][0].cpu().tolist(),grasp_distance_mm=float((s['tip'][0,0]-site[0]).norm()*1000),
                        jaws_rad=s['jaws'][0].cpu().tolist(),forces_n=s['forces'][0].norm(dim=-1).cpu().tolist(),
                        force_vectors_n=s['forces'][0].cpu().tolist(),peg_quaternion=s['peg_q'][0].cpu().tolist(),
                        tool_quaternions=s['tool_q'][0].cpu().tolist(),contact_valid=s['contact_valid'][0].cpu().tolist(),
                        action=executed[0].cpu().tolist(),stage=int(teacher.stage[0]) if teacher else None))
                    if net is not None and hasattr(net,'outputs'):
                        logits,heads=net.outputs(x)
                        choice=net.selected_option if getattr(net,'selected_option',None) is not None else logits.argmax(-1)
                        probe[-1]['learned_option']=int(choice[0])
                        goals=heads[0,choice[0]].reshape(2,8).clone()
                        if getattr(net,'selected_goal',None) is not None and bool(net.latch_option_poses):
                            goals[:,:7]=net.selected_goal[0].reshape(2,8)[:,:7]
                        if getattr(net,'current_goal',None) is not None:
                            goals=net.current_goal[0].reshape(2,8).clone()
                        if getattr(net,'jaw_rate_head',None) is not None:
                            reference=x[0,[34,69]]*.5
                            if hasattr(net,'jaw_rates'):
                                rates=net.jaw_rates(x,choice)[0]
                            else:
                                decoder=net.pickup_decoder if getattr(net,'pickup_decoder',None) is not None and int(choice[0])<6 else net
                                jaw_choice=choice.clamp_max(5) if decoder is not net else choice
                                rates=decoder.jaw_rate_head(x,jaw_choice)[0]
                            goals[:,7]=4*(reference+rates*.15*.01).clamp(0,.5)-1
                        probe[-1]['learned_tool_goals']=goals.cpu().tolist()
                obs,_,done,infos=vec.step(executed.cpu().numpy());step+=1
                if teacher:teacher.update()
                ids=torch.as_tensor(np.flatnonzero(done),device=args.device)
                for i in ids.cpu().tolist():
                    ep=int(episode_id[i]);row=infos[i]['metrics'];row['reset']=conditions[ep]
                    terminal=physical_observation(torch.as_tensor(infos[i]['terminal_observation'],device=args.device))
                    row['terminal_physics']=dict(peg_m=(terminal[70:73]*.1).cpu().tolist(),
                        bottom_m=float(terminal[83]*.05),tool_tips_m=[(terminal[:3]*.1).cpu().tolist(),(terminal[35:38]*.1).cpu().tolist()],
                        forces_n=[terminal[21:27].reshape(2,3).norm(dim=-1).cpu().tolist(),terminal[56:62].reshape(2,3).norm(dim=-1).cpu().tolist()],
                        jaws_rad=[(terminal[19:21]*.5).cpu().tolist(),(terminal[54:56]*.5).cpu().tolist()])
                    row['terminal_teacher_stage']=int(stage[i]) if teacher else None
                    if net is not None and getattr(net,'selected_option',None) is not None:
                        row['terminal_learned_option']=int(net.selected_option[i])
                    if record_allocated_result(completed,ep,row,args.episodes):
                        print(f'[AUTONOMY {args.mode.upper()}] {len(completed)}/{args.episodes} episode={ep} '
                              f'success={row["success"]} grasp={row["grasp_psm1"]} peak={row["peak_force_n"]:.3f}N',flush=True)
                    episode_id[i]=next_id;conditions[next_id]=dict(vec.reset_records[i]);next_id+=1
                if teacher and len(ids):teacher.reset(ids)
                if len(ids):previous_learner[ids]=False
                if net is not None and hasattr(net,'reset') and len(ids):net.reset(ids)
                if step%500==0:
                    write(args.output/'motion_probe.json',probe)
                    write(args.output/'live_progress.json',dict(complete=False,step=step,episodes=len(completed),
                        progress=vec.progress.flags.sum(0).cpu().tolist(),teacher_at_inference=teacher is not None,
                        stages=teacher.stage.cpu().tolist() if teacher else None,
                        learned_options=net.selected_option.cpu().tolist() if getattr(net,'selected_option',None) is not None else None,
                        physical_observations=x.cpu().tolist(),
                        learned_goals=net.current_goal.cpu().tolist() if getattr(net,'current_goal',None) is not None else None,
                        decision_scores=net.decision_scores.nan_to_num(neginf=-1e9,posinf=1e9).cpu().tolist() if getattr(net,'decision_scores',None) is not None else None,
                        forces_n=vec.state['forces'].norm(dim=-1).cpu().tolist()))
                    write(args.output/'completed_episodes.json',dict(complete=False,requested_episodes=args.episodes,
                        episodes=len(completed),cases=list(completed.values()),teacher_at_inference=teacher is not None))
                    successful_count=sum(row['success'] for row in completed.values())
                    if teacher and successful_count>saved_successes:
                        save_completed_demonstrations(args.output,chunks,completed,conditions,manifest,SCHEMA,args,partial=True)
                        saved_successes=successful_count
                    site=vec.state['peg']+apply(vec.state['peg_q'],local.expand(args.num_envs,-1))
                    distance=(vec.state['tip'][:,0]-site).norm(dim=-1)*1000
                    print(f'[AUTONOMY] step={step} episodes={len(completed)}/{args.episodes} '
                        f'progress={vec.progress.flags.sum(0).cpu().tolist()} distance_mm={distance[:4].cpu().tolist()}'+
                        (f' stages={torch.bincount(teacher.stage,minlength=19 if args.skill!="lift" else 6).cpu().tolist()}' if teacher else ''),flush=True)
                if args.realtime:time.sleep(max(0,.01-(time.monotonic()-tick)))
                if not app.is_running():raise KeyboardInterrupt
        rows=[completed[ep] for ep in sorted(completed)]
        report=dict(complete=True,episodes=len(rows),success=sum(r['success'] for r in rows),grasp_psm1=sum(r['grasp_psm1'] for r in rows),
            sampling='Fixed allocated episode IDs; every allocated episode completes without replacement by faster environments',
            lift_stable=sum(r['lift_stable'] for r in rows),grasp_psm2=sum(r['grasp_psm2'] for r in rows),
            handover=sum(r['handover'] for r in rows),full_success=sum(r['full_success'] for r in rows),
            timeouts=sum(r['timeout'] for r in rows),failures=sum(r['failure'] for r in rows),cases=rows,
            assisted_grasp=bool(args.grasp_support),programmed_task_controller=teacher is not None,teacher_at_inference=teacher is not None,
            mode=args.mode,policy_sha256=manifest['policy_sha256'],seed=args.seed,skill=args.skill,gpu=torch.cuda.get_device_name(),
            motion_reference=vec.reference.settings(),
            motion_timing=dict(rollout_wall_seconds=time.monotonic()-rollout_started,realtime_requested=args.realtime,
                successful_episode_seconds=[r['length']*.01 for r in rows if r['success']]),
            reset_distribution=dict(sources=args.sources,targets=args.targets,jitter_mm=args.jitter_mm,yaw_deg=args.yaw_deg,
                translation_sampling='Uniform XY square centered on the same fixed source post; half-width in mm',
                rotation_sampling='Uniform yaw about the peg hole center; bounds in degrees',reset_only=True),
            handover_criterion='Receiver alone supports peg above 37.5 mm for 2 s, with stable opposed contacts and donor force below 0.01 N',
            low_level_control='IK with generic contact speed limits; no object attachment or task goals')
        if args.grasp_support:
            from autonomy_support import SUPPORT_PROFILES
            settings=SUPPORT_PROFILES[args.support_profile]
            report.update(native_full_policy_verified=False,grasp_support=dict(type='bounded_grasp_spring',strength=args.grasp_support,
                profile=args.support_profile,force_limit_n=settings['force_limit_n']*args.grasp_support,
                torque_limit_nm=settings['torque_limit_nm']*args.grasp_support,
                placement_guide_enabled=args.placement_support,placement_force_limit_n=.15*args.grasp_support,
                placement_torque_limit_nm=.002*args.grasp_support,object_pose_writes=False,object_velocity_writes=False),
                handover_criterion='Measured opposed receiver contacts maintained for 2s with donor fingers free; virtual spring assistance remains explicitly active',
                low_level_control='Neural tool motors through generic IK/speed limits, plus explicit grasp spring and optional free-placement guide')
        if any(row['assisted_grasp']!=report['assisted_grasp'] for row in rows):raise ValueError('Inconsistent assistance evidence')
        if teacher:
            metadata=save_completed_demonstrations(args.output,chunks,completed,conditions,manifest,SCHEMA,args)
            report['dataset']=metadata
        write(args.output/'evaluation.json',report)
        write(args.output/'live_progress.json',dict(complete=True,episodes=report['episodes'],success=report['success'],
            full_success=report['full_success'],report='evaluation.json'))
        write(args.output/'motion_probe.json',probe)
        print(json.dumps({k:v for k,v in report.items() if k not in ('cases','dataset')},indent=2),flush=True)
        return int(report['success']<args.episodes*.9)
    except BaseException as error:
        write(args.output/'error.json',dict(type=type(error).__name__,message=str(error)));raise
    finally:
        try:
            if vec is not None:vec.close()
            elif env is not None:env.close()
            evaluation=args.output/'evaluation.json'
            write(args.output/'environment_close.json',dict(environment_closed=True,
                evaluation_complete=evaluation.is_file(),evaluation_sha256=sha(evaluation) if evaluation.is_file() else None))
        except BaseException as error:
            write(args.output/'environment_close.json',dict(environment_closed=False,error=f'{type(error).__name__}: {error}'))
            raise
        finally:
            app.close()

if __name__=='__main__':raise SystemExit(main())
