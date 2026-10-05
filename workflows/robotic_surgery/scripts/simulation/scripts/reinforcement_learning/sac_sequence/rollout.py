"""Shared user-run baseline, demonstration collection, evaluation and inference."""
import hashlib
import json
import traceback
import numpy as np
from common import SCHEMA, TASK, assets, checkpoint, cli, fingerprint, make_env, versions

METRICS = ('grasp_psm1','lift_stable','grasp_psm2','handover','placement','full_success')

def interval(k,n):
    if not n: return [0.,1.]
    z=1.959963984540054; p=k/n; d=1+z*z/n
    center=(p+z*z/(2*n))/d
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,center-radius),min(1.,center+radius)]

def run(mode):
    args = cli(mode)
    meta = checkpoint(args.checkpoint) if args.checkpoint else None
    if meta and meta['source'] != args.source:
        raise ValueError('Use the trained fixed source in this initial curriculum')
    if args.guidance is None:
        args.guidance = meta['guidance'] if meta else 1.
    if mode == 'collect':
        args.baseline = True
        if args.checkpoint: raise ValueError('Collection uses the analytic teacher, not a legacy network')
    from isaaclab.app import AppLauncher
    app, env, trace_file = None,None,None
    episodes, transitions, demo_episodes = [],[],[]
    requested = args.episodes * (len(args.guidance_levels) if mode == 'collect' else 1)
    interrupted = False
    args.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        app = AppLauncher(args).app
        env, cfg = make_env(args)
        from robotic.surgery.tasks.surgical.peg_transfer_sequence import mdp
        from robotic.surgery.tasks.surgical.peg_transfer_sequence import assistance
        from robotic.surgery.tasks.surgical.peg_transfer import sac_mdp as physical
        from stable_baselines3 import SAC
        model = SAC.load(args.checkpoint,env=env,device=args.device) if meta else None
        if args.trace or mode == 'probe':
            trace_file = args.output.with_suffix('.trace.jsonl').open('x')
        levels = args.guidance_levels if mode == 'collect' else [args.guidance]
        rng = np.random.default_rng(args.seed)
        requested = args.episodes*len(levels)
        total_steps = 0
        for level in levels:
            cfg.guidance = level
            obs = env.reset()
            completed = 0
            case_rows = []
            last_stages = None
            while completed < args.episodes:
                if not app.is_running():
                    interrupted=True; break
                before = mdp.state(env.unwrapped).stage.detach().cpu().numpy().copy()
                stages = tuple(before.tolist())
                if stages != last_stages or total_steps % 100 == 0:
                    ages = mdp.state(env.unwrapped).age.tolist()
                    print(f'[SEQUENCE {mode.upper()}] step={total_steps} '
                          f'stages={[mdp.STAGES[min(int(s),19)] for s in before]} '
                          f'age={ages} holder={physical.state(env.unwrapped).holder.tolist()}',flush=True)
                    last_stages = stages
                teacher = mdp.expert_action(env.unwrapped).detach().cpu().numpy().copy()
                action = teacher if args.baseline else model.predict(obs,deterministic=True)[0]
                if mode == 'collect' and args.action_noise:
                    action = np.clip(action+rng.normal(0,args.action_noise,action.shape),-1,1).astype(np.float32)
                if trace_file:
                    base = env.unwrapped
                    record = dict(step=total_steps,stage=before.tolist(),
                        stage_names=[mdp.STAGES[min(int(s),19)] for s in before],
                        guidance=level,action=action.tolist(),expert_action=teacher.tolist(),
                        tips_m=mdp.tips(base).tolist(),targets_m=mdp.goals(base).tolist(),
                        tool_quaternions_w=[base.scene[f'ee_{a}_frame'].data.target_quat_w[:,0].tolist() for a in (1,2)],
                        robot_joint_positions=[base.scene[f'robot_{a}'].data.joint_pos.tolist() for a in (1,2)],
                        peg_root_m=physical.local_peg(base).tolist(),
                        peg_quaternion_w=base.scene['object'].data.root_quat_w.tolist(),
                        assistance=assistance.telemetry(base),
                        jaws_rad=np.stack([physical.jaw(base,a).cpu().numpy() for a in (1,2)],axis=1).tolist(),
                        metrics={k:getattr(physical.state(base),k).tolist() for k in METRICS})
                    trace_file.write(json.dumps(record)+'\n')
                next_obs,reward,done,infos = env.step(action)
                if mode == 'collect':
                    terminal = infos[0]['terminal_observation'] if done[0] else next_obs[0]
                    case_rows.append(dict(observations=obs[0].copy(),actions=action[0].copy(),bc_actions=teacher[0].copy(),
                        next_observations=terminal.copy(),rewards=float(reward[0]),dones=float(done[0]),
                        stages=int(before[0]),guidance=float(level)))
                obs=next_obs;total_steps+=1
                for i in np.flatnonzero(done):
                    if completed >= args.episodes: break
                    result=dict(**infos[i]['transfer'],**infos[i]['episode'],
                                timeout=infos[i]['TimeLimit.truncated'],guidance=level)
                    episodes.append(result); completed+=1
                    with args.output.with_suffix('.episodes.jsonl').open('a') as f:
                        f.write(json.dumps(result)+'\n')
                    print(f'[SEQUENCE {mode.upper()}] {len(episodes)}/{requested}: '
                          f'stage={result["sequence"]["stage_name"]} success={result["phase_success"]}',flush=True)
                    if mode == 'collect':
                        if result['metrics']['full_success'] and not result['failure']:
                            start=len(transitions);transitions.extend(case_rows)
                            demo_episodes.append(dict(start=start,stop=len(transitions),full_success=True,
                                guidance=level,metrics=result['metrics']))
                        case_rows=[]
            if interrupted: break
        complete=not interrupted and len(episodes)==requested
        report=dict(task=TASK,schema=SCHEMA,fingerprint=fingerprint(),assets=assets(),versions=versions(),checkpoint=str(args.checkpoint.resolve()) if meta else None,
            source=args.source,target='R2',skill=args.skill,guidance=levels,baseline_controller=args.baseline,
            receiver_grasp_offset_m=list(mdp.g.GRASP2),receiver_approach='positive_y_edge',
            learned_active_motor_without_guidance=bool(meta and all(level==0 for level in levels)),
            sequencer_and_inactive_arm_control=True,assisted_grasp=True,physical_grasp_verified=False,
            episodes=len(episodes),requested_episodes=requested,complete=complete,interrupted=interrupted,
            seed=args.seed,num_envs=args.num_envs,episode_seconds=args.episode_seconds,
            skill_success=sum(e['phase_success'] for e in episodes),
            timeouts=sum(e['timeout'] for e in episodes),failures=sum(e['failure'] for e in episodes),
            blocked_moves=sum(e['blocked_moves'] for e in episodes),
            terminal_stages={stage:sum(e['sequence']['stage_name']==stage for e in episodes)
                             for stage in (*mdp.STAGES,'complete')},
            terminal_examples=[e['sequence'] for e in episodes[:6]])
        report['skill_success_rate']=report['skill_success']/len(episodes) if episodes else 0.
        report['skill_success_ci95']=interval(report['skill_success'],len(episodes))
        for name in ('transfer_zone_reached','reach_psm2','donor_released','insertion_reached'):
            report[name]=sum(e['sequence'][name] for e in episodes)
            report[name+'_rate']=report[name]/len(episodes) if episodes else 0.
        for name in METRICS:
            count=sum(e['metrics'][name] for e in episodes)
            report[name]=count;report[name+'_rate']=count/len(episodes) if episodes else 0.
            report[name+'_ci95']=interval(count,len(episodes))
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(f'[SEQUENCE] Report: {args.output}',flush=True)
        if trace_file: print(f'[SEQUENCE] Trace: {args.output.with_suffix(".trace.jsonl")}',flush=True)
        print(json.dumps(report,indent=2),flush=True)
        if mode == 'collect':
            if not complete or len(demo_episodes)!=requested:
                print('[COLLECT] Archive not saved: every requested full physical transfer must succeed',flush=True)
            else:
                from model import load_demos
                fields=tuple(transitions[0])
                arrays={k:np.asarray([row[k] for row in transitions],dtype=np.float32 if k not in ('stages',) else np.int64)
                        for k in fields}
                args.demonstrations.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(args.demonstrations,**arrays)
                archive_meta=dict(schema=SCHEMA+'-demonstrations',task=TASK,fingerprint=fingerprint(),assets=assets(),
                    receiver_grasp_offset_m=list(mdp.g.GRASP2),receiver_approach='positive_y_edge',
                    source=args.source,target='R2',episode_seconds=args.episode_seconds,step_dt=env.unwrapped.step_dt,
                    transitions=len(transitions),observation_dim=arrays['observations'].shape[1],episodes=demo_episodes,
                    guidance_levels=levels,action_noise=args.action_noise,assisted_grasp=True,physical_grasp_verified=False,
                    sha256=hashlib.sha256(args.demonstrations.read_bytes()).hexdigest())
                args.demonstrations.with_suffix('.json').write_text(json.dumps(archive_meta,indent=2)+'\n')
                load_demos(args.demonstrations)
                print(f'[COLLECT] Verified demonstrations: {args.demonstrations}',flush=True)
    except KeyboardInterrupt:
        partial=dict(task=TASK,schema=SCHEMA,fingerprint=fingerprint(),source=args.source,target='R2',
            skill=args.skill,guidance=args.guidance,baseline_controller=args.baseline,complete=False,
            interrupted=True,episodes=len(episodes),requested_episodes=requested,
            note='Interrupted evaluation: no gate may use this report')
        args.output.write_text(json.dumps(partial,indent=2)+'\n')
        print(f'[SEQUENCE] Interrupted; partial report: {args.output}',flush=True)
    except Exception:
        traceback.print_exc();raise
    finally:
        if trace_file: trace_file.close()
        if env is not None: env.close()
        if app is not None: app.close()
