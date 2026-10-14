"""Frozen BC + bounded residual SAC, using checkpoint13 physics and controls."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path.cwd().resolve()
HERE = Path(__file__).resolve().parent
FILES = ('residual_main.py', 'residual_env.py', 'residual_learning.py','residual_grasp_guard.py')
PARENT = 'milestones/checkpoint13-three-posts-fast-assisted.json'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value): Path(path).write_text(json.dumps(value, indent=2) + '\n')


def diagnostic(vec, step, out):
    """Measured state and the BC's own goal, before executing or resetting."""
    import torch
    from rl_core import canonical,rotvec,multiply,conjugate
    raw=vec.venv;state=raw.state
    with torch.no_grad():
        goals=vec.base.current_goal.reshape(-1,2,8)
        error=(goals[...,:3]*.1-state['tip']).norm(dim=-1)*1000
        q=canonical(goals[...,3:7]/goals[...,3:7].norm(dim=-1,keepdim=True).clamp_min(1e-6))
        rotation=rotvec(multiply(q,conjugate(state['tool_q']))).norm(dim=-1)
        row=dict(step=step,age=raw.age.tolist(),options=vec.base.selected_option.tolist(),
            goals=goals.tolist(),goal_error_mm=error.tolist(),rotation_error_rad=rotation.tolist(),
            tool_tip_m=state['tip'].tolist(),tool_q=state['tool_q'].tolist(),peg_m=state['peg'].tolist(),peg_q=state['peg_q'].tolist(),
            jaws=state['jaws'].tolist(),reference_jaw=raw.reference.jaw.tolist(),
            forces_n=state['forces'].norm(dim=-1).tolist(),flags=raw.progress.flags.tolist(),holder=raw.support.holder.tolist(),
            base_actions=vec.next_base.tolist(),reference_speed_mm_s=(raw.reference.speed_cap*1000).tolist(),
            joints=state['joints'].tolist(),held_jaw_guard=vec.held_jaw_guard.evidence() if vec.held_jaw_guard else None,
            decision_scores=vec.base.decision_scores.nan_to_num(neginf=-1e9,posinf=1e9).tolist())
    with (out/'diagnostic.trace.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
    write(out/'live_progress.json',row)
    print(f'[RESIDUAL DIAG] step={step} options={row["options"]} receiver_error_mm={[round(x[1],3) for x in row["goal_error_mm"]]}',flush=True)


def verify_files(folder, hashes):
    for name, digest in hashes.items():
        if sha(folder / name) != digest: raise ValueError(f'Artifact changed: {name}')


def stage(args):
    """Execute the experiment from its own snapshot, never mutable BC sources."""
    if args.snapshot_ready: return
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.checkpoint:
        candidate = args.checkpoint.resolve().parent
        saved = json.loads((candidate / 'manifest.json').read_text())
        verify_files(candidate / 'sources', saved['source_hashes'])
        verify_files(candidate / 'base', saved['base_hashes'])
        if sha(args.checkpoint) != saved['sac_sha256']: raise ValueError('SAC weights changed')
        if sha(ROOT/'milestones/checkpoint3-full-assisted.json') != saved['physics_manifest_sha256']:
            raise ValueError('Candidate physics manifest changed')
        # Physics/runtime/mixing must match the experiment. Evaluation distribution may vary explicitly.
        args.translation = saved['residual_limits']['translation']
        args.held_jaw_guard=saved.get('held_jaw_guard_enabled',False)
        source = candidate / 'sources'
        base = candidate / 'base'
    else:
        source = HERE
        parent = json.loads((ROOT / PARENT).read_text())
        base = ROOT / parent['preserved_bundle'] / 'model'
        verify_files(ROOT / parent['preserved_bundle'], parent['files'])
        if sha(ROOT / parent['physics_manifest']) != parent['physics_manifest_sha256']:
            raise ValueError('Original physics manifest changed')
    shutil.copytree(base, out / 'base', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (out / 'sources').mkdir()
    for name in FILES: shutil.copy2(source / name, out / 'sources' / name)
    argv = [sys.executable, '-u', str(out / 'sources/residual_main.py'), *sys.argv[1:],
            '--output', str(out), '--translation', str(args.translation), '--snapshot_ready']
    argv=[item for item in argv if item!='--held_jaw_guard']
    if args.held_jaw_guard:argv.append('--held_jaw_guard')
    os.execv(sys.executable, argv)


def evaluate(vec, model, args, out):
    import numpy as np
    import torch
    obs = vec.reset()
    ids = np.arange(vec.num_envs); next_id = vec.num_envs; completed = {}
    started = time.monotonic()
    for step in range(1, args.episode_steps * (1 + (args.episodes - 1) // vec.num_envs) + 1):
        tick = time.monotonic()
        if args.diagnostic_interval and (step-1)%args.diagnostic_interval==0:diagnostic(vec,step-1,out)
        vec.active = torch.as_tensor(ids < args.episodes, device=vec.device)
        residual = np.zeros((vec.num_envs, 14), np.float32) if model is None else model.predict(obs, deterministic=True)[0]
        obs, _, dones, infos = vec.step(residual)
        for i in np.flatnonzero(dones):
            if ids[i] < args.episodes:
                row = infos[i]['metrics'].copy(); row['episode_id'] = int(ids[i])
                completed[int(ids[i])] = row
                print(f'[RESIDUAL EVAL] {len(completed)}/{args.episodes} pair={row["reset"]} success={row["success"]}', flush=True)
            ids[i] = next_id; next_id += 1
        if step % 500 == 0:
            print(f'[RESIDUAL EVAL] step={step} complete={len(completed)}/{args.episodes} progress={vec.venv.progress.flags.sum(0).tolist()}', flush=True)
            write(out/'status.json',dict(complete=False,stage='physical_evaluation',step=step,
                episodes=len(completed),requested_episodes=args.episodes,success=sum(r['success'] for r in completed.values())))
        if len(completed) == args.episodes: break
        if args.realtime: time.sleep(max(0, .01 - (time.monotonic() - tick)))
    rows = [completed[k] for k in sorted(completed)]
    report = dict(complete=len(rows) == args.episodes, episodes=len(rows), requested_episodes=args.episodes,
        success=sum(r['success'] for r in rows), full_success=sum(r['full_success'] for r in rows),
        failures=sum(r['failure'] for r in rows), timeouts=sum(r['timeout'] for r in rows),
        episodes_detail=rows, seed=args.seed, wall_seconds=time.monotonic()-started,
        algorithm='Frozen BC + residual SAC' if model is not None else 'Frozen BC, exactly zero residual',
        base_policy_sha256=sha(out/'base/policy.pt'), sac_sha256=sha(args.checkpoint) if args.checkpoint else None,
        reset_distribution=dict(sources=['L5','L6','L4'], targets=['R2','R1','R3'], paired=True,
                                jitter_mm=args.jitter_mm, yaw_deg=args.yaw_deg),
        motion_reference=vec.venv.reference.settings(), **vec.evidence())
    report['provenance_sha256']=sha(out/'provenance.json')
    write(out/'evaluation.json', report)
    return report


def train(vec, args, out):
    import numpy as np
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from residual_learning import make_model, warm_critic
    model = make_model(vec, args.seed, args.learning_rate, args.buffer_size, args.batch_size)
    obs = vec.reset(); completed = {}; warm_steps = 0
    # First real trajectories are the frozen BC in THESE assisted dynamics, not old native demos.
    for step in range(1, args.episode_steps + 1):
        if args.diagnostic_interval and (step-1)%args.diagnostic_interval==0:diagnostic(vec,step-1,out)
        action = np.zeros((vec.num_envs, 14), np.float32)
        nxt, reward, dones, infos = vec.step(action)
        terminal = nxt.copy()
        for i in np.flatnonzero(dones):
            terminal[i] = infos[i]['terminal_observation']
            completed.setdefault(int(i), infos[i]['metrics'].copy())
        model.replay_buffer.add(obs, terminal, action, reward, dones, infos)
        obs = nxt; warm_steps = step * vec.num_envs
        if step % 500 == 0:
            print(f'[RESIDUAL WARM] steps={step} first_episodes={len(completed)}/{vec.num_envs}', flush=True)
            write(out/'status.json', dict(complete=False, stage='bc_warm_rollouts', transitions=warm_steps,
                first_episodes=len(completed), successes=sum(r['success'] for r in completed.values())))
        if len(completed) == vec.num_envs: break
    if len(completed) != vec.num_envs or not all(row['success'] for row in completed.values()):
        write(out/'status.json', dict(complete=True, stage='baseline_gate_failed', first_episodes=completed))
        raise RuntimeError('Zero-residual baseline failed: no SAC optimization or checkpoint promotion')
    write(out/'warm_rollouts.json', dict(transitions=warm_steps, first_episodes=completed))
    frozen_before = {k:v.detach().clone() for k,v in vec.base.state_dict().items()}
    warm_critic(model, args.critic_updates)
    print(f'[RESIDUAL SAC] critic warmed; {args.transitions} online transitions requested', flush=True)

    class ProgressCallback(BaseCallback):
        def _on_step(self):
            if self.n_calls % 250 == 0:
                print(f'[RESIDUAL SAC] {self.num_timesteps}/{args.transitions} transitions', flush=True)
                write(out/'status.json', dict(complete=False, stage='sac_training',
                    transitions=self.num_timesteps, requested=args.transitions, warm_transitions=warm_steps))
            return True

    model.learn(args.transitions, callback=ProgressCallback())
    if any(not torch.equal(value, vec.base.state_dict()[key]) for key,value in frozen_before.items()):
        raise RuntimeError('Frozen BC parameters were changed')
    model.save(out/'candidate.zip')
    meta = dict(schema=vec.evidence()['schema'], parent=PARENT, base_policy_sha256=sha(out/'base/policy.pt'),
        held_jaw_guard_enabled=args.held_jaw_guard,
        sac_sha256=sha(out/'candidate.zip'), source_hashes={name:sha(out/'sources'/name) for name in FILES},
        physics_manifest_sha256=sha(ROOT/'milestones/checkpoint3-full-assisted.json'),
        base_hashes={str(p.relative_to(out/'base')):sha(p) for p in (out/'base').rglob('*') if p.is_file() and '__pycache__' not in p.parts},
        residual_limits=vec.evidence()['residual_limits'], training_parameters=vars(args)|{'output':str(out),'checkpoint':None},
        algorithm='Frozen structured BC + anchored residual SAC; free translation curriculum',
        bc_weights_unchanged=True, gamma=.9995, tau=.005, entropy_coefficient=1e-5,
        anchor_weight=model.anchor_weight, exploration_log_std_target=-5., actor_critic_architecture=[256,256],
        assisted_grasp=True, teacher_at_inference=False, native_full_policy_verified=False,
        promoted=False, physical_evaluation_required=True)
    write(out/'manifest.json', meta)
    write(out/'status.json', dict(complete=True, stage='candidate_requires_physical_evaluation',
        transitions=model.num_timesteps, candidate=str(out/'candidate.zip'), sac_sha256=meta['sac_sha256']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('baseline','train','evaluate','play'))
    p.add_argument('--output', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--snapshot_ready', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--num_envs', type=int, default=6); p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--episode_steps', type=int, default=10000); p.add_argument('--seed', type=int, default=61001)
    p.add_argument('--transitions', type=int, default=12000); p.add_argument('--critic_updates', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=512); p.add_argument('--buffer_size', type=int, default=100000)
    p.add_argument('--learning_rate', type=float, default=3e-5); p.add_argument('--translation', type=float, default=.05)
    p.add_argument('--jitter_mm', type=float, default=0.); p.add_argument('--yaw_deg', type=float, default=0.)
    p.add_argument('--diagnostic_interval',type=int,default=100)
    p.add_argument('--held_jaw_guard',action='store_true',help='Explicit temporary aperture support after measured stable grasp')
    p.add_argument('--headless', action='store_true'); p.add_argument('--realtime', action='store_true')
    p.add_argument('--video', action='store_true')
    a = p.parse_args()
    if a.output is None: a.output = ROOT/'logs/grasp_v2'/f'residual_{a.mode}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if a.checkpoint: a.checkpoint = a.checkpoint.resolve()
    if a.mode in ('evaluate','play') and not a.checkpoint: p.error('Specify --checkpoint candidate.zip')
    if a.mode in ('baseline','train') and a.checkpoint: p.error('Start from frozen checkpoint13')
    if min(a.num_envs,a.episodes,a.episode_steps,a.batch_size,a.transitions,a.critic_updates,a.buffer_size)<1: p.error('Positive counts required')
    if not 1e-6<=a.learning_rate<=1e-3: p.error('Learning rate must be finite and within 1e-6..1e-3')
    if a.diagnostic_interval<0:p.error('Diagnostic interval must be nonnegative; zero disables tracing')
    if not 0<=a.jitter_mm<=1 or not 0<=a.yaw_deg<=5: p.error('Initial curriculum bounds: 0..1mm and 0..5deg')
    if a.video and (a.mode not in ('evaluate','play') or a.num_envs!=1 or a.episodes!=1): p.error('Video requires one evaluation/play episode')
    stage(a)
    out = a.output.resolve()
    write(out/'provenance.json',dict(parent=PARENT,base_policy_sha256=sha(out/'base/policy.pt'),
        source_hashes={name:sha(out/'sources'/name) for name in FILES},
        base_hashes={str(f.relative_to(out/'base')):sha(f) for f in (out/'base').rglob('*') if f.is_file() and '__pycache__' not in f.parts},
        physics_manifest_sha256=sha(ROOT/'milestones/checkpoint3-full-assisted.json'),
        parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}))
    sys.path.insert(0, str(out/'base/sources'))
    import replay_full
    replay_full.REPO=ROOT
    base, bundle = replay_full.verify(ROOT/'milestones/checkpoint3-full-assisted.json')
    scripts = bundle/'workflows/robotic_surgery/scripts'; ext=scripts/'simulation/exts'
    sys.path[:0]=[str(scripts),str(ext/'robotic.surgery.tasks'),str(ext/'robotic.surgery.assets')]
    from isaaclab.app import AppLauncher
    app = AppLauncher(argparse.Namespace(headless=a.headless or a.video, enable_cameras=a.video, device='cuda:0')).app
    vec = env = recorder = None
    try:
        import torch
        torch.set_num_threads(8); torch.manual_seed(a.seed)
        from rl_runtime import make_env, initialize_contact_physics
        from autonomy_env import AutonomyEnv
        from autonomy_learning import load_policy
        from autonomy_support import GraspSpring
        from autonomy_gripper_guard import GripperLoadGuard
        from autonomy_upright_support import install
        from residual_env import ResidualEnv, ResidualLimits
        physical = argparse.Namespace(**base['parameters']); physical.num_envs=a.num_envs; physical.seed=a.seed; physical.device='cuda:0'
        physical.robot_usd=bundle/'data/Isaac/Healthcare/0.5.0/132c82d/Robots/dVRK/PSM/psm.usd'
        profile=json.loads((bundle/'tools/peg_grasp_v2/peg_profile.json').read_text())
        env=make_env(physical,profile); initialize_contact_physics(env,physical.friction)
        env.sim.set_camera_view(eye=(.025,.32,.25),target=(.025,0.,.035))
        if a.video:
            from replay import EpisodeVideo
            recorder=EpisodeVideo(env,out/'inference_front.mp4')
        raw=AutonomyEnv(env,profile,'full',a.episode_steps,a.seed,a.jitter_mm,a.yaw_deg,
                       ['L5','L6','L4'],['R2','R1','R3'],4.,1.,'fast_travel',8.)
        env.profile=profile; raw.support=GraspSpring(env,1.,'firm',True); raw.progress.grasp_support=raw.support
        raw.paired_posts=True; install(raw)
        from residual_grasp_guard import HeldJawGuard
        vec=ResidualEnv(raw,load_policy(out/'base/policy.pt','cuda:0'),GripperLoadGuard(a.num_envs,'cuda:0'),ResidualLimits(a.translation),
                        HeldJawGuard(a.num_envs,'cuda:0') if a.held_jaw_guard else None)
        if a.mode=='train': train(vec,a,out)
        else:
            from residual_learning import ResidualSAC
            model=ResidualSAC.load(a.checkpoint,env=vec,device='cuda:0') if a.checkpoint else None
            report=evaluate(vec,model,a,out)
            write(out/'status.json',dict(complete=report['complete'],stage='evaluation_complete',success=report['success'],episodes=report['episodes']))
        vec.close(); vec=None; env=None
        write(out/'environment_close.json',dict(complete=True,closed=True))
        if recorder:
            import subprocess
            info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                '-show_entries','stream=codec_name,pix_fmt,nb_read_frames,width,height,duration','-of','json',str(out/'inference_front.mp4')]))
            stream=info['streams'][0]
            subprocess.run(['ffmpeg','-v','error','-i',str(out/'inference_front.mp4'),'-f','null','-'],check=True)
            if int(stream['nb_read_frames'])!=recorder.frames: raise ValueError('Video frame count mismatch')
            expected_steps=sum(row['length'] for row in report['episodes_detail'])
            if abs(recorder.frames/25-expected_steps*.01)>.05:
                raise ValueError('Video does not cover the full physical episode')
            write(out/'video.json',dict(video_sha256=sha(out/'inference_front.mp4'),sac_sha256=sha(a.checkpoint),
                base_policy_sha256=sha(out/'base/policy.pt'), report_sha256=sha(out/'evaluation.json'),
                source_hashes={name:sha(out/'sources'/name) for name in FILES},
                camera=dict(eye=[.025,.32,.25],target=[.025,0.,.035]), stream=stream,
                complete_episode=report['complete'],full_success=report['full_success'],decode_verified=True))
        return 0
    except Exception as error:
        write(out/'error.json',dict(error=repr(error),complete=False))
        raise
    finally:
        if vec is not None: vec.close()
        elif env is not None: env.close()
        app.close()


if __name__=='__main__': raise SystemExit(main())
