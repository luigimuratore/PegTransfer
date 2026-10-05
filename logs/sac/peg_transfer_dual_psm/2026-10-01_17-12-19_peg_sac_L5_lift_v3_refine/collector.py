"""Short, user-run Isaac Sim probe of source reach, jaw closure and assisted lift.

Uses the real IK actions and jaws. It never teleports a tool/peg, changes grasp
thresholds, or labels a failed trajectory as successful. It is not a learned policy.
No simulator imports occur until after CLI parsing (including --help).
"""

import argparse
import json
import math
from pathlib import Path
import sys

import torch


def probe_limits(stage_steps=None, episode_seconds=None, step_dt=0.02, approach_mode='preclose'):
    stages = [('settle', 25), ('above_side', 150), ('outside_side', 150),
              ('reach', 100), ('close', 100), ('lift', 150), ('hold', 100)]
    if approach_mode == 'preclose':
        stages.insert(3, ('preclose', 100))
    if stage_steps is not None:
        stages = [(name, count if name == 'settle' else stage_steps) for name, count in stages]
    duration = episode_seconds if episode_seconds is not None else max(20., sum(n for _, n in stages) * step_dt + 2.)
    return stages, duration


def approach_target(grasp_w, approach_mode):
    # Stop inside the existing 4 mm capture radius, outside the collider surface.
    # Advancing all the way to the nominal point with open jaws disturbed the peg.
    offset = (0., -0.0035, 0.) if approach_mode in ('preclose', 'capture') else (0., 0., 0.)
    return grasp_w + grasp_w.new_tensor(offset)


def stage_achieved(stage, row, approach_mode):
    if stage == 'preclose':
        return row['jaw_closed']
    if stage == 'reach' and approach_mode == 'preclose':
        return row['holder'] == 1 or (row['near'] and row['aligned'] and row['jaw_closed'])
    if stage == 'reach' and approach_mode == 'capture':
        return row['near'] and row['aligned'] and row['jaw_mean_rad'] > 0.38 and row['holder'] == 0
    if stage in ('above_side', 'outside_side', 'reach'):
        return row['target_error_m'] < 0.002
    if stage == 'close':
        return row['holder'] == 1
    if stage == 'lift':
        return row['peg_xyz'][2] >= 0.058
    return row['metrics']['lift_stable']


def closure_evidence(rows):
    """Evidence of open arrival followed by closed capture, not contact forces."""
    reach = [r for r in rows if r['stage'] == 'reach' and not r['reset']]
    close = [r for r in rows if r['stage'] == 'close' and not r['reset']]
    arrival = reach[-1] if reach else None
    started = bool(arrival and stage_achieved('reach', arrival, 'capture'))
    completed = started and any(r['near'] and r['aligned'] and r['jaw_closed']
                                and r['holder'] == 1 for r in close)
    return dict(closure_started_in_capture_zone=started,
                closure_completed_in_capture_zone=completed)


def cartesian_action(env, target_w, jaw_command, max_translation=0.0012):
    """Conservative proportional reach command using the task's actual IK scale."""
    action = torch.zeros(env.num_envs, 14, device=env.device)
    tip = env.scene['ee_1_frame'].data.target_pos_w[:, 0]
    delta = (target_w - tip) * 0.5
    delta *= (max_translation / delta.norm(dim=1).clamp_min(max_translation))[:, None]
    q = env.scene['robot_1'].data.root_quat_w
    # Rotate world-frame displacement by the inverse root quaternion (wxyz).
    xyz = -q[:, 1:]
    uv = torch.linalg.cross(xyz, delta, dim=1)
    delta_b = delta + 2 * (q[:, :1] * uv + torch.linalg.cross(xyz, uv, dim=1))
    scale = env.action_manager.get_term('body_1_joint_pos').cfg.scale
    action[:, :3] = delta_b / action.new_tensor(scale[:3])
    action[:, 6] = jaw_command
    action[:, 13] = 1.0  # PSM2 stays open; both rotational commands remain zero.
    return action.clamp(-1, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    scripts = Path(__file__).resolve().parents[1] / 'reinforcement_learning/sac'
    sys.path.insert(0, str(scripts))
    from common import SOURCE_LABELS
    p.add_argument('--source', choices=('both', 'near', 'random', *SOURCE_LABELS), default='both')
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--output', type=Path, default=Path('logs/sac/grasp_probe.json'))
    p.add_argument('--phase', choices=('full', 'lift'), default='full')
    p.add_argument('--demonstrations', type=Path,
                   help='Write real observation/action/reward/terminal transitions to a new .npz archive')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--stage_steps', type=int, default=None,
                   help='Maximum control steps per movement/closure/hold stage; 50 steps = 1 s')
    p.add_argument('--episode_seconds', type=float, default=None,
                   help='Episode timeout in simulation seconds; default covers all stage budgets plus 2 s')
    p.add_argument('--approach_mode', choices=('preclose', 'capture', 'open'), default='preclose',
                   help='capture: arrive open, stop, close; preclose: close outside then approach; open: original sequence')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--enable_cameras', action='store_true')
    args = p.parse_args()
    if args.repeats < 1:
        p.error('--repeats must be positive')
    if args.stage_steps is not None and args.stage_steps < 1:
        p.error('--stage_steps must be positive')
    if args.episode_seconds is not None and (not math.isfinite(args.episode_seconds) or args.episode_seconds <= 0):
        p.error('--episode_seconds must be finite and positive')
    if args.demonstrations:
        if args.phase != 'lift' or args.approach_mode != 'capture' or args.episode_seconds is not None:
            p.error('Demonstrations require --phase lift --approach_mode capture, without --episode_seconds overrides')
        if args.demonstrations.suffix != '.npz':
            p.error('--demonstrations must end with .npz')
        if args.demonstrations.exists() or args.demonstrations.with_suffix('.json').exists():
            p.error('Demonstration output exists; choose a new filename to preserve it')
    from common import TASK, fingerprint, make_env, versions
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    env, trace, cases = None, [], []
    demo_rows, demo_episodes = [], []
    gripper_names = None
    requested = args.source
    args.num_envs, args.video = 1, False
    sources = ('L5', 'L6') if requested in ('both', 'near') else (
        SOURCE_LABELS if requested == 'random' else (requested,))
    args.source = sources[0]
    complete = False
    stage_limits, episode_seconds = probe_limits(args.stage_steps, args.episode_seconds, approach_mode=args.approach_mode)
    try:
        from robotic.surgery.tasks.surgical.peg_transfer import sac_geometry as g, sac_mdp as mdp
        env, _ = make_env(args, args.phase)
        base = env.unwrapped
        stage_limits, episode_seconds = probe_limits(args.stage_steps, args.episode_seconds, base.step_dt, args.approach_mode)
        # Isaac Lab max_episode_length is a property of cfg.episode_length_s/step_dt.
        # Apply before reset; the diagnostic is separate from learned-policy evaluation.
        if args.demonstrations:
            episode_seconds = base.cfg.episode_length_s
        else:
            base.cfg.episode_length_s = episode_seconds
        print(f'[PROBE] Episode: {episode_seconds:g} s / {base.max_episode_length} steps; stages={stage_limits}', flush=True)
        gripper_ids, gripper_names = base.scene['robot_1'].find_joints('psm_tool_gripper.*_joint')
        for source in sources:
            for repetition in range(args.repeats):
                if not app.is_running():
                    break
                base.cfg.source_post = source
                previous_obs = env.reset()
                case_transitions = []
                origin = base.scene.env_origins
                source_root = mdp.state(base).source.clone()
                source_root[:, 2] = g.REST_Z
                grasp_w = source_root + source_root.new_tensor(g.GRASP1) + origin
                close_target = None
                steps, ended, stage_records = 0, False, []
                for stage, limit in stage_limits:
                    stable, achieved = 0, False
                    reason = None
                    stage_rows = []
                    for _ in range(limit):
                        if not app.is_running():
                            ended = True
                            reason = 'window_closed'
                            break
                        if stage == 'settle':
                            target = base.scene['ee_1_frame'].data.target_pos_w[:, 0].clone()
                        elif stage == 'above_side':
                            target = grasp_w + grasp_w.new_tensor((0., -0.015, 0.020))
                        elif stage in ('outside_side', 'preclose'):
                            target = grasp_w + grasp_w.new_tensor((0., -0.015, 0.))
                        elif stage in ('reach', 'close'):
                            target = (close_target if stage == 'close' and close_target is not None
                                      else approach_target(grasp_w, args.approach_mode))
                        else:
                            root_goal = source_root.clone()
                            root_goal[:, 2] = 0.060
                            target = root_goal + origin - mdp.state(base).offset
                        closed = stage in ('preclose', 'close', 'lift', 'hold') or (
                            stage == 'reach' and args.approach_mode == 'preclose')
                        max_translation = 0.0003 if stage == 'reach' and args.approach_mode in ('preclose', 'capture') else 0.0012
                        action = cartesian_action(base, target, -1.0 if closed else 1.0, max_translation)
                        obs, reward, done, infos = env.step(action)
                        steps += 1
                        if args.demonstrations:
                            next_obs = infos[0]['terminal_observation'] if bool(done[0]) else obs[0]
                            case_transitions.append(dict(observations=previous_obs[0].copy(),
                                actions=action[0].detach().cpu().numpy().copy(),
                                next_observations=next_obs.copy(), rewards=float(reward[0]), dones=float(done[0]),
                                sources=source, stages=stage))
                        previous_obs = obs
                        if bool(done[0]):
                            ended = True
                            terminal = infos[0]['transfer']
                            reason = ('failure' if terminal['failure'] else
                                      'success' if terminal['phase_success'] else
                                      'time_out' if infos[0]['TimeLimit.truncated'] else 'other_reset')
                            row = dict(source=source, repetition=repetition, step=steps, stage=stage,
                                       reset=True, termination_reason=reason, terminal=terminal)
                            trace.append(row)
                            achieved = args.phase == 'lift' and terminal['phase_success'] and terminal['metrics']['lift_stable']
                            break
                        s, data = mdp.state(base), base.scene['object'].data
                        peg, tip = mdp.local_peg(base)[0], mdp.ee(base, 1)[0] - origin[0]
                        distance = float((tip - peg - peg.new_tensor(g.GRASP1)).norm())
                        qnorm = float(data.root_quat_w[0, 1:].norm())
                        measured_jaw = float(mdp.jaw(base, 1)[0])
                        row = dict(source=source, repetition=repetition, step=steps, stage=stage,
                                   reset=False, grasp_distance_m=distance, jaw_mean_rad=measured_jaw,
                                   jaw_closed=measured_jaw < 0.18, near=distance < g.GRASP_RADIUS,
                                   aligned=qnorm < 0.005, quaternion_vector_norm=qnorm,
                                   orientation_error_deg=math.degrees(2 * math.asin(min(1., qnorm))),
                                   candidate_steps=int(s.candidate1[0]), holder=int(s.holder[0]),
                                   blocked_moves=int(s.blocked_moves[0]), peg_xyz=peg.tolist(), tip_xyz=tip.tolist(),
                                   target_error_m=float((target[0] - mdp.ee(base, 1)[0]).norm()),
                                   gripper_joints=base.scene['robot_1'].data.joint_pos[0, gripper_ids].tolist(),
                                   metrics={k: bool(getattr(s, k)[0]) for k in mdp.METRICS})
                        trace.append(row)
                        stage_rows.append(row)
                        if stage == 'settle':
                            continue
                        condition = stage_achieved(stage, row, args.approach_mode)
                        stable = stable + 1 if condition else 0
                        dwell = 5 if stage in ('above_side', 'outside_side', 'preclose') or (
                            stage == 'reach' and args.approach_mode == 'open') else 1
                        if stable >= dwell:
                            achieved = True
                            if stage == 'reach' and args.approach_mode in ('preclose', 'capture'):
                                # Hold the measured tool pose during attachment dwell;
                                # do not keep pushing towards the nominal grasp point.
                                close_target = mdp.ee(base, 1).clone()
                            break
                    if stage == 'settle':
                        achieved = not ended
                    if reason is None:
                        reason = 'achieved' if achieved else 'stage_budget_exhausted'
                    stage_steps = len(stage_rows) + int(ended and reason != 'window_closed')
                    summary = dict(stage=stage, achieved=achieved, steps=stage_steps,
                                   step_limit=limit, termination_reason=reason)
                    if stage_rows:
                        summary.update(min_grasp_distance_m=min(r['grasp_distance_m'] for r in stage_rows),
                                       min_target_error_m=min(r['target_error_m'] for r in stage_rows),
                                       min_jaw_mean_rad=min(r['jaw_mean_rad'] for r in stage_rows),
                                       near_closed_aligned_steps=sum(r['near'] and r['jaw_closed'] and r['aligned'] for r in stage_rows),
                                       last=stage_rows[-1])
                    stage_records.append(summary)
                    print(f'[PROBE] {source} {stage}: achieved={achieved}; steps={stage_steps}; reason={reason}', flush=True)
                    if ended or not achieved:
                        break
                metrics = next((r.get('metrics', r.get('terminal', {}).get('metrics'))
                                for r in reversed(trace) if r['source'] == source and r['repetition'] == repetition), None)
                evidence = closure_evidence([r for r in trace if r['source'] == source and r['repetition'] == repetition])
                passed = bool(metrics and metrics['lift_stable'])
                if args.approach_mode == 'capture':
                    passed &= evidence['closure_completed_in_capture_zone']
                cases.append(dict(source=source, repetition=repetition, stages=stage_records, steps=steps,
                                  metrics=metrics, **evidence, passed=passed))
                if args.demonstrations and passed and case_transitions and case_transitions[-1]['dones']:
                    start = len(demo_rows)
                    demo_rows.extend(case_transitions)
                    demo_episodes.append(dict(source=source, repetition=repetition, start=start, stop=len(demo_rows),
                                              passed=True, metrics=metrics, **evidence))
        complete = len(cases) == len(sources) * args.repeats
    except KeyboardInterrupt:
        print('[PROBE] Interrupted; saving available measurements', flush=True)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = dict(task=TASK, fingerprint=fingerprint(), versions=versions(), seed=args.seed,
                      controlled_probe=True, learned_policy=False, complete=complete, cases=cases,
                      probe_revision=4, approach_mode=args.approach_mode, phase=args.phase,
                      validation_scope='scripted_assisted_grasp_and_lift', physical_grasp_verified=False,
                      episode_seconds=episode_seconds, stage_limits=dict(stage_limits),
                      passed=complete and all(c['passed'] for c in cases),
                      gripper_joint_names=gripper_names if env is not None else None,
                      thresholds=dict(grasp_radius_m=0.004, closed_jaw_rad=0.18, aligned_qnorm=0.005))
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        args.output.with_suffix('.trace.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in trace))
        print(f'[PROBE] Report: {args.output}; passed={report["passed"]}', flush=True)
        try:
            if args.demonstrations:
                if report['passed'] and len(demo_episodes) == len(cases):
                    from demonstrations import save_demonstrations
                    meta = save_demonstrations(args.demonstrations, demo_rows, demo_episodes, episode_seconds, base.step_dt)
                    print(f'[PROBE] Demonstrations: {args.demonstrations}; transitions={meta["transitions"]}', flush=True)
                else:
                    print('[PROBE] Demonstrations not saved: collection must complete with every case successful', flush=True)
        finally:
            if env is not None:
                env.close()
            app.close()


if __name__ == '__main__':
    main()
