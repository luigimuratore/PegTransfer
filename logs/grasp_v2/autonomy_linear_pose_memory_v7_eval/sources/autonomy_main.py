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

def snapshot(out):
    folder=out/'sources';folder.mkdir(parents=True,exist_ok=True)
    names=['rl_core.py','rl_env.py','rl_runtime.py','geometry.py','peg_profile.json','replay_full.py','replay.py']
    for path in list(HERE.glob('autonomy_*.py'))+[HERE/name for name in names]:shutil.copy2(path,folder/path.name)
    return {p.name:sha(p) for p in folder.iterdir() if p.is_file()}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('collect','train','evaluate','play','repackage'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--policy',type=Path)
    p.add_argument('--datasets',type=Path,nargs='+');p.add_argument('--num_envs',type=int,default=16)
    p.add_argument('--episodes',type=int,default=32);p.add_argument('--episode_steps',type=int,default=5000)
    p.add_argument('--jitter_mm',type=float,default=.5);p.add_argument('--yaw_deg',type=float,default=2.)
    p.add_argument('--sources',nargs='+',default=['L5']);p.add_argument('--targets',nargs='+',default=['R2'])
    p.add_argument('--learner_fraction',type=float,default=0.,help='DAgger probability of executing the learner per environment/step')
    p.add_argument('--learner_until_stage',type=int,default=19,help='Limit DAgger perturbations to approach stages while collecting initial recoveries')
    p.add_argument('--updates',type=int,default=6000);p.add_argument('--batch_size',type=int,default=1024)
    p.add_argument('--architecture',choices=('mlp','neighbors','gru','pose','mixture','linear_options'),default='mlp')
    p.add_argument('--future_steps',type=int,default=0,help='BC target from future measured poses; no future observations at inference')
    p.add_argument('--phase_endpoint',action='store_true',help='For pose BC, label measured stage endpoints offline; stage is NOT an actor input')
    p.add_argument('--skill',choices=('lift','handover','full'),default='lift')
    p.add_argument('--balance_direction',type=float,choices=(-1.,1.),default=1.,help='Collection-only receiver contact centering experiment')
    p.add_argument('--implementation_change',default='Preserve XYZ/rotation direction when bounding long pose commands',help='Description of the current-code fork when repackaging unchanged weights')
    p.add_argument('--transition_data',type=Path,nargs='+',help='Repackage: infer option-transition memory from complete training demonstrations')
    p.add_argument('--latch_option_poses',action='store_true',help='Repackage: choose and retain a learned pose on each learned option change; jaws remain closed-loop learned outputs')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda:0')
    p.add_argument('--headless',action='store_true');p.add_argument('--realtime',action='store_true');p.add_argument('--video',action='store_true')
    args=p.parse_args()
    if not args.device.startswith('cuda'):p.error('CUDA required')
    if min(args.num_envs,args.episodes,args.episode_steps,args.updates,args.batch_size)<1:p.error('Positive counts required')
    if not 0<=args.learner_fraction<=1:p.error('learner_fraction must be 0..1')
    if not 0<=args.future_steps<=100:p.error('future_steps must be 0..100')
    if args.phase_endpoint and args.architecture!='pose':p.error('phase_endpoint requires architecture pose')
    if args.jitter_mm<0 or args.jitter_mm>1 or not 0<=args.yaw_deg<=5:p.error('First curriculum: jitter 0..1 mm and yaw 0..5 deg')
    if (args.mode in ('evaluate','play') or args.learner_fraction>0) and not args.policy:p.error('Specify --policy')
    if args.video and (args.num_envs!=1 or args.episodes!=1):p.error('Video needs one environment and episode')
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
        if args.transition_data:
            from autonomy_learning import demonstrated_transitions
            if not hasattr(net,'allowed_transitions'):raise ValueError('Transition memory requires option actor')
            allowed,initial,counts=demonstrated_transitions(args.transition_data,net.options,args.device)
            net.allowed_transitions.copy_(allowed);net.initial_option.fill_(initial)
            transition_details=dict(transition_counts=counts.cpu().tolist(),initial_option=initial,
                transition_dataset_hashes={str(path):sha(path) for path in args.transition_data})
        if args.latch_option_poses:
            if not hasattr(net,'latch_option_poses'):raise ValueError('Pose memory requires option actor')
            net.latch_option_poses.fill_(True)
            transition_details['latch_option_poses']=True
        with torch.inference_mode():
            action=net(net.mean[None])
            if action.shape!=(1,14) or not bool(torch.isfinite(action).all()):raise ValueError('Invalid actor')
        args.output.mkdir(parents=True,exist_ok=False)
        if args.transition_data or args.latch_option_poses:
            saved=torch.load(args.policy,map_location=args.device,weights_only=True)
            saved['state_dict']=net.state_dict();torch.save(saved,args.output/'policy.pt')
        else:shutil.copy2(args.policy,args.output/'policy.pt')
        cfg=json.loads((args.policy.parent/'training.json').read_text())
        cfg.update(retrained=False,neural_parameters_unchanged=True,implementation_change=args.implementation_change,**transition_details,
                   parent_policy=str(args.policy.resolve()),physical_evaluation_required=True)
        write(args.output/'training.json',cfg)
        parent.update(source_hashes=snapshot(args.output),parent_checkpoint=str(args.policy.resolve()),policy_sha256=sha(args.output/'policy.pt'),
            parent_policy_sha256=sha(args.policy),**transition_details,
            implementation_change=cfg['implementation_change'],retrained=False)
        write(args.output/'manifest.json',parent)
        return 0
    if args.mode=='train':
        if not args.datasets:p.error('Specify datasets')
        for dataset in args.datasets:
            with np.load(dataset,allow_pickle=False) as archive:
                if json.loads(str(archive['metadata']))['skill']!=args.skill:
                    p.error('Dataset skill must match --skill')
        # Execute the snapshot, not files that might be edited during a long
        # training run. The preserved source must match the experiment code.
        with tempfile.TemporaryDirectory(prefix='peg_autonomy_train_',dir='/tmp') as frozen:
            hashes=snapshot(Path(frozen));folder=Path(frozen)/'sources'
            sys.path.insert(0,str(folder))
            from autonomy_learning import train,SCHEMA
            train(args.datasets,args.output,args.device,args.updates,args.batch_size,args.seed,args.architecture,args.future_steps,args.phase_endpoint)
            shutil.copytree(folder,args.output/'sources')
            write(args.output/'manifest.json',dict(schema=SCHEMA,source_hashes=hashes,policy_sha256=sha(args.output/'policy.pt'),
                dataset_hashes={str(path):sha(path) for path in args.datasets},
                physics_manifest_sha256=sha(REPO/'milestones/checkpoint3-full-assisted.json'),
                skill=args.skill,assisted_grasp=False,programmed_task_controller=False))
        return 0
    from replay_full import verify
    base,bundle=verify(REPO/'milestones/checkpoint3-full-assisted.json')
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        source_hashes=snapshot(args.output),physics_manifest_sha256=sha(REPO/'milestones/checkpoint3-full-assisted.json'),
        assisted_grasp=False,programmed_task_controller=args.mode=='collect',skill=args.skill,
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
        vec=AutonomyEnv(env,profile,args.skill,args.episode_steps,args.seed,args.jitter_mm,args.yaw_deg,args.sources,args.targets)
        vec.balance_direction=args.balance_direction
        if args.skill!='lift':
            from autonomy_full_teacher import FullTeacher
            teacher_type=FullTeacher
        else:teacher_type=PickupTeacher
        obs=vec.reset();teacher=teacher_type(vec) if args.mode=='collect' else None
        net=load_policy(args.policy,args.device) if args.policy else None
        chunks=[];episode_id=torch.arange(args.num_envs,device=args.device);next_id=args.num_envs
        completed={};conditions={int(i):dict(vec.reset_records[int(i)]) for i in range(args.num_envs)};step=0
        probe=[];local=torch.tensor(contact_candidate(profile,'wall',2,.5,1)['tip_local_m'],device=args.device,dtype=torch.float32)
        with torch.inference_mode():
            while len(vec.episodes)<args.episodes:
                tick=time.monotonic();x=physical_observation(torch.as_tensor(obs,device=args.device))
                if teacher:
                    label=teacher.action();stage=teacher.stage.clone();executed=label
                    if net is not None:
                        use=(torch.rand(args.num_envs,device=args.device)<args.learner_fraction)&(stage<args.learner_until_stage)
                        executed=torch.where(use[:,None],net(x),label)
                    chunks.append(dict(obs=x.clone(),actions=label.clone(),stage=stage,episode=episode_id.clone()))
                else:executed=net(x)
                if step%50==0:
                    s=vec.state;site=s['peg']+apply(s['peg_q'],local.expand(args.num_envs,-1))
                    probe.append(dict(step=step,episode=int(episode_id[0]),tip_m=s['tip'][0].cpu().tolist(),
                        peg_m=s['peg'][0].cpu().tolist(),grasp_distance_mm=float((s['tip'][0,0]-site[0]).norm()*1000),
                        jaws_rad=s['jaws'][0].cpu().tolist(),forces_n=s['forces'][0].norm(dim=-1).cpu().tolist(),
                        action=executed[0].cpu().tolist(),stage=int(teacher.stage[0]) if teacher else None))
                    if net is not None and hasattr(net,'outputs'):
                        logits,heads=net.outputs(x)
                        choice=net.selected_option if getattr(net,'selected_option',None) is not None else logits.argmax(-1)
                        probe[-1]['learned_option']=int(choice[0])
                        goals=heads[0,choice[0]].reshape(2,8).clone()
                        if getattr(net,'selected_goal',None) is not None and bool(net.latch_option_poses):
                            goals[:,:7]=net.selected_goal[0].reshape(2,8)[:,:7]
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
                    completed[ep]=row
                    print(f'[AUTONOMY {args.mode.upper()}] {len(completed)}/{args.episodes} episode={ep} '
                          f'success={row["success"]} grasp={row["grasp_psm1"]} peak={row["peak_force_n"]:.3f}N',flush=True)
                    episode_id[i]=next_id;conditions[next_id]=dict(vec.reset_records[i]);next_id+=1
                if teacher and len(ids):teacher.reset(ids)
                if net is not None and hasattr(net,'reset') and len(ids):net.reset(ids)
                if step%500==0:
                    write(args.output/'motion_probe.json',probe)
                    write(args.output/'live_progress.json',dict(complete=False,step=step,episodes=len(vec.episodes),
                        progress=vec.progress.flags.sum(0).cpu().tolist(),teacher_at_inference=teacher is not None,
                        stages=teacher.stage.cpu().tolist() if teacher else None,
                        forces_n=vec.state['forces'].norm(dim=-1).cpu().tolist()))
                    site=vec.state['peg']+apply(vec.state['peg_q'],local.expand(args.num_envs,-1))
                    distance=(vec.state['tip'][:,0]-site).norm(dim=-1)*1000
                    print(f'[AUTONOMY] step={step} episodes={len(vec.episodes)}/{args.episodes} '
                        f'progress={vec.progress.flags.sum(0).cpu().tolist()} distance_mm={distance[:4].cpu().tolist()}'+
                        (f' stages={torch.bincount(teacher.stage,minlength=19 if args.skill!="lift" else 6).cpu().tolist()}' if teacher else ''),flush=True)
                if args.realtime:time.sleep(max(0,.01-(time.monotonic()-tick)))
                if not app.is_running():raise KeyboardInterrupt
        rows=vec.episodes[:args.episodes]
        report=dict(complete=True,episodes=len(rows),success=sum(r['success'] for r in rows),grasp_psm1=sum(r['grasp_psm1'] for r in rows),
            lift_stable=sum(r['lift_stable'] for r in rows),grasp_psm2=sum(r['grasp_psm2'] for r in rows),
            handover=sum(r['handover'] for r in rows),full_success=sum(r['full_success'] for r in rows),
            timeouts=sum(r['timeout'] for r in rows),failures=sum(r['failure'] for r in rows),cases=rows,
            assisted_grasp=False,programmed_task_controller=teacher is not None,teacher_at_inference=teacher is not None,
            mode=args.mode,policy_sha256=manifest['policy_sha256'],seed=args.seed,skill=args.skill,gpu=torch.cuda.get_device_name(),
            low_level_control='IK with generic contact speed limits; no object attachment or task goals')
        if teacher:
            joined={k:torch.cat([c[k] for c in chunks]).cpu().numpy() for k in chunks[0]}
            successful=[k for k,v in completed.items() if v['success']]
            keep=np.isin(joined['episode'],successful)
            data={k:v[keep] for k,v in joined.items()}
            # Reserve every fifth successful complete episode for offline validation.
            validation_episodes=successful[4::5] or successful[-1:]
            data['validation']=np.isin(data['episode'],validation_episodes)
            metadata=dict(schema=SCHEMA,skill=args.skill,assisted_grasp=False,teacher=True,learner_fraction=args.learner_fraction,
                episodes=successful,validation_episodes=validation_episodes,transitions=int(keep.sum()),parameters=manifest['parameters'],
                resets={str(k):conditions[k] for k in successful},source_hashes=manifest['source_hashes'])
            with (args.output/'demonstrations.npz').open('xb') as f:np.savez_compressed(f,**data,metadata=json.dumps(metadata))
            report['dataset']=metadata
        write(args.output/'evaluation.json',report)
        write(args.output/'motion_probe.json',probe)
        print(json.dumps({k:v for k,v in report.items() if k not in ('cases','dataset')},indent=2),flush=True)
        return int(report['success']<args.episodes*.9)
    except BaseException as error:
        write(args.output/'error.json',dict(type=type(error).__name__,message=str(error)));raise
    finally:
        if vec is not None:vec.close()
        elif env is not None:env.close()
        app.close()

if __name__=='__main__':raise SystemExit(main())
