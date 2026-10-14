"""Train or inspect a free-action SAC policy in the preserved v2 contact scene."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

REPO=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')

def run_evaluation(model,env,episodes,realtime=False):
    import numpy as np
    from rl_core import DT
    obs=env.reset();start=len(env.episodes);steps=0
    while len(env.episodes)-start<episodes:
        tick=time.monotonic();action,_=model.predict(obs,deterministic=True)
        obs,_,_,_=env.step(action);steps+=1
        if steps%500==0:
            print(f'[FREE SAC EVAL] step={steps} episodes={len(env.episodes)-start}/{episodes} '
                  f'physical_progress={env.progress.flags.sum(0).cpu().tolist()}',flush=True)
        if realtime:time.sleep(max(0,DT-(time.monotonic()-tick)))
    rows=env.episodes[start:start+episodes]
    return dict(episodes=episodes,success=sum(r['success'] for r in rows),
        full_success=sum(r['full_success'] for r in rows),
        grasp_psm1=sum(r['grasp_psm1'] for r in rows),lift_stable=sum(r['lift_stable'] for r in rows),
        grasp_psm2=sum(r['grasp_psm2'] for r in rows),handover=sum(r['handover'] for r in rows),
        failures=sum(r['failure'] for r in rows),timeouts=sum(r['timeout'] for r in rows),
        cases=rows,complete=True,deterministic=True,assisted_grasp=False,
        programmed_task_controller=False,stage_waypoints=False,learned_motor_actions=14)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['train','evaluate','play','smoke'])
    parser.add_argument('--demonstrations',type=Path,default=REPO/'logs/grasp_v2/sac_free_demonstrations_v2.npz')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--skill',choices=['lift','handover','full'])
    parser.add_argument('--num_envs',type=int,default=16)
    parser.add_argument('--steps',type=int,default=100000)
    parser.add_argument('--episode_steps',type=int,default=8000)
    parser.add_argument('--bc_steps',type=int,default=6000)
    parser.add_argument('--critic_warmup',type=int,default=1000)
    parser.add_argument('--batch_size',type=int,default=1024)
    parser.add_argument('--learning_rate',type=float,default=1e-4)
    parser.add_argument('--gamma',type=float,default=.9995)
    parser.add_argument('--ent_coef',type=float,default=1e-5)
    parser.add_argument('--bc_weight',type=float,default=1.)
    parser.add_argument('--demo_fraction',type=float,default=.25)
    parser.add_argument('--gradient_steps',type=int,default=1)
    parser.add_argument('--episodes',type=int,default=3)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--realtime',action='store_true')
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--skip_pre_eval',action='store_true')
    parser.add_argument('--promotion_report',type=Path)
    parser.add_argument('--buffer_size',type=int,default=250000)
    args=parser.parse_args()
    args.skill=args.skill or ('lift' if args.mode=='train' else 'full')
    if args.gamma!=.9995:parser.error('Demonstration rewards use gamma=.9995; re-export before changing gamma')
    if args.num_envs<1 or args.episodes<1 or args.episode_steps<1:parser.error('Positive environment and episode counts required')
    if args.video and args.num_envs!=1:parser.error('Video requires --num_envs 1')
    if args.mode in ('evaluate','play') and args.checkpoint is None:parser.error('Specify --checkpoint')
    if args.mode=='train' and args.skill!='lift' and args.checkpoint is None:
        parser.error('Curriculum: initialize lift first, then resume a validated checkpoint')
    if not args.device.startswith('cuda'):parser.error('This experiment requires CUDA')
    from replay_full import verify
    base,bundle=verify(REPO/'milestones/checkpoint3-full-assisted.json')
    stamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    out=(args.output or REPO/f'logs/grasp_v2/sac_free_{args.skill}_{args.mode}_{stamp}').resolve()
    out.mkdir(parents=True,exist_ok=False)
    # Every new checkpoint carries its exact controller/reward/training sources.
    (out/'sources').mkdir()
    for path in HERE.glob('rl_*.py'):shutil.copy2(path,out/'sources'/path.name)
    shutil.copy2(bundle/'tools/peg_grasp_v2/geometry.py',out/'sources/geometry.py')
    shutil.copy2(bundle/'tools/peg_grasp_v2/peg_profile.json',out/'sources/peg_profile.json')
    params={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    manifest=dict(schema='peg-free-sac-v2-1',parameters=params,assisted_grasp=False,
        actor='14 signed continuous XYZ/angular-velocity/jaw actions, both arms',
        actor_inputs='110 physical state/reference/history features; final destination; no stage or waypoint',
        curriculum_skill=args.skill,source='L5',target='R2',domain_randomization=False,
        physics_base_manifest=str(REPO/'milestones/checkpoint3-full-assisted.json'),
        base_manifest_sha256=sha(REPO/'milestones/checkpoint3-full-assisted.json'),versions=base['versions'],
        limitation='First experiment, physical rollout evaluation required; receiver grip previously slipped without fixture',
        source_hashes={p.name:sha(p) for p in (out/'sources').iterdir() if p.is_file()})
    if args.checkpoint:
        parent=json.loads((args.checkpoint.parent/'manifest.json').read_text())
        if parent['schema']!=manifest['schema']:raise ValueError('Incompatible checkpoint')
        for name,value in parent['source_hashes'].items():
            source=HERE/name if name!='geometry.py' else bundle/'tools/peg_grasp_v2/geometry.py'
            if not source.is_file() or sha(source)!=value:
                raise ValueError('Use checkpoint frozen sources when replaying, or explicitly migrate/re-export the experiment')
        manifest['parent_checkpoint']=str(args.checkpoint.resolve());manifest['parent_checkpoint_sha256']=sha(args.checkpoint)
        previous=parent['curriculum_skill']
        if args.mode=='train' and previous!=args.skill:
            order=['lift','handover','full']
            if order.index(args.skill)!=order.index(previous)+1:raise ValueError('Advance one curriculum skill at a time')
            gate_path=args.promotion_report or args.checkpoint.parent/'evaluation.json'
            gate=json.loads(gate_path.read_text())
            if not gate.get('complete') or gate.get('assisted_grasp') or gate['episodes']<16 or gate['success']/gate['episodes']<.90:
                raise ValueError('Curriculum gate: at least 16 episodes, >=90% measured success without assistance required')
            if any(r['skill']!=previous for r in gate['cases']):raise ValueError('Wrong skill in curriculum gate report')
            if gate.get('checkpoint_sha256')!=sha(args.checkpoint):raise ValueError('Curriculum gate must evaluate these exact checkpoint weights')
            manifest['curriculum_gate']=dict(report=str(gate_path),sha256=sha(gate_path))
    write(out/'manifest.json',manifest)
    scripts=bundle/'workflows/robotic_surgery/scripts';ext=scripts/'simulation/exts'
    sys.path[:0]=[str(scripts),str(ext/'robotic.surgery.tasks'),str(ext/'robotic.surgery.assets')]
    from isaaclab.app import AppLauncher
    launch=argparse.Namespace(headless=args.headless or args.video,device=args.device,enable_cameras=args.video)
    launcher=AppLauncher(launch);app=launcher.app;physics=None;vec=None
    try:
        import torch
        torch.set_num_threads(8)
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        from rl_runtime import make_env,initialize_contact_physics
        from rl_env import FreePegVecEnv
        from rl_learning import FreeSAC,CUDAReplayBuffer,StandardFeatures,load_data,initialize_actor
        from stable_baselines3 import SAC
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.callbacks import BaseCallback
        profile=json.loads((bundle/'tools/peg_grasp_v2/peg_profile.json').read_text())
        physical=argparse.Namespace(**base['parameters'])
        physical.num_envs=args.num_envs;physical.device=args.device;physical.seed=args.seed
        physical.robot_usd=bundle/'data/Isaac/Healthcare/0.5.0/132c82d/Robots/dVRK/PSM/psm.usd'
        physics=make_env(physical,profile);initialize_contact_physics(physics,physical.friction)
        physics.sim.set_camera_view(eye=(.025,.32,.25),target=(.025,0.,.035))
        if args.video:
            from replay import EpisodeVideo
            EpisodeVideo(physics,out/'inference_front.mp4')
        vec=FreePegVecEnv(physics,profile,args.skill,args.episode_steps,args.gamma)
        manifest['gpu_name']=torch.cuda.get_device_name();manifest['actual_materials']=physics.grasp_material_evidence
        write(out/'manifest.json',manifest)
        if args.mode=='smoke':
            import numpy as np
            obs=vec.reset()
            for _ in range(100):obs,_,_,_=vec.step(np.zeros((args.num_envs,14),dtype=np.float32))
            if not np.isfinite(obs).all():raise ValueError('Invalid physical observation')
            write(out/'smoke.json',dict(passed=True,device=args.device,observations=list(obs.shape),steps=100,num_envs=args.num_envs))
            print('[FREE SAC] CUDA physics smoke passed',flush=True);return 0
        if args.mode=='train':
            data,meta=load_data(args.demonstrations,args.device)
            if meta.get('gamma')!=args.gamma:raise ValueError('Demonstration reward discount does not match this experiment')
            if args.checkpoint:
                model=FreeSAC.load(args.checkpoint,env=vec,device=args.device)
                model.replay_buffer.configure_demos(args.demonstrations,args.skill)
                model.learning_rate=args.learning_rate
                model.gradient_steps=args.gradient_steps;model.batch_size=args.batch_size
                model.replay_buffer.demo_fraction=args.demo_fraction
                model.bc_data=data;model.bc_indices=(data['episode']!=meta['validation_episode']).nonzero().flatten()
            else:
                model=FreeSAC('MlpPolicy',vec,device=args.device,learning_rate=args.learning_rate,
                    buffer_size=args.buffer_size,learning_starts=0,batch_size=args.batch_size,
                    gamma=args.gamma,tau=.005,train_freq=1,gradient_steps=args.gradient_steps,
                    ent_coef=args.ent_coef,seed=args.seed,verbose=0,replay_buffer_class=CUDAReplayBuffer,
                    replay_buffer_kwargs=dict(demonstrations=str(args.demonstrations),demo_fraction=args.demo_fraction,skill=args.skill),
                    policy_kwargs=dict(net_arch=[256,256],features_extractor_class=StandardFeatures,
                        features_extractor_kwargs=dict(mean=meta['mean'],std=meta['std'])))
                model.set_logger(configure(str(out),['csv']))
                manifest['bc_validation']=initialize_actor(model,data,meta,args.bc_steps,args.batch_size)
            model.set_logger(configure(str(out),['csv']));model.bc_weight=args.bc_weight
            model.save(out/'pretrained.zip');manifest['demonstrations_sha256']=sha(args.demonstrations)
            write(out/'manifest.json',manifest)
            if not args.skip_pre_eval:
                before=run_evaluation(model,vec,args.episodes)
                before['checkpoint_sha256']=sha(out/'pretrained.zip')
                write(out/'pretrained_evaluation.json',before)
                print(f'[FREE SAC] Before RL: {before["success"]}/{before["episodes"]} {args.skill}; grasp {before["grasp_psm1"]}',flush=True)
            model.replay_buffer.only_demo=True
            optimizer=model.actor.optimizer
            optimizer.step=lambda *a,**kw:None
            try:
                for k in range(args.critic_warmup):
                    SAC.train(model,1,args.batch_size)
                    if (k+1)%250==0:print(f'[FREE SAC] Critic warmup {k+1}/{args.critic_warmup}',flush=True)
            finally:del optimizer.step;model.replay_buffer.only_demo=False
            class ProgressLog(BaseCallback):
                def __init__(self):super().__init__();self.last=0
                def _on_step(self):
                    if self.num_timesteps-self.last>=5000:
                        self.last=self.num_timesteps
                        recent=vec.episodes[-32:]
                        print(f'[FREE SAC] {self.num_timesteps}/{args.steps} transitions; '
                              f'episodes={len(vec.episodes)} recent_success={sum(x["success"] for x in recent)}/{len(recent)}',flush=True)
                        self.model.save(out/'latest.zip')
                    return True
            try:model.learn(args.steps,callback=ProgressLog(),reset_num_timesteps=True)
            except KeyboardInterrupt:
                manifest['interrupted']=True;print('[FREE SAC] Interrupted; saving current policy',flush=True)
            model.save(out/'final.zip');manifest['checkpoint_sha256']=sha(out/'final.zip')
            write(out/'training_episodes.json',vec.episodes)
            result=run_evaluation(model,vec,args.episodes)
        else:
            model=FreeSAC.load(args.checkpoint,env=vec,device=args.device)
            result=run_evaluation(model,vec,args.episodes,args.realtime)
        result['checkpoint_sha256']=sha(out/'final.zip' if args.mode=='train' else args.checkpoint)
        write(out/'evaluation.json',result)
        manifest['evaluation']=dict(success=result['success'],episodes=result['episodes'],skill=args.skill,full_success=result['full_success'])
        write(out/'manifest.json',manifest)
        print(json.dumps(dict(output=str(out),evaluation=manifest['evaluation'],assisted_grasp=False),indent=2),flush=True)
        return 0
    except BaseException as error:
        write(out/'error.json',dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        if vec is not None:vec.close()
        elif physics is not None:physics.close()
        app.close()

if __name__=='__main__':raise SystemExit(main())
