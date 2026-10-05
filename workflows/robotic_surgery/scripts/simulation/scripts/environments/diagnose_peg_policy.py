"""One deterministic policy episode with measured grasp telemetry, run by the user.

This observer never commands an expert action or changes task physics/rewards.
Its location outside the SAC source bundle preserves existing checkpoint hashes.
Samples are at control frequency (50 Hz), not every physics substep (200 Hz).
"""

import argparse
import json
from pathlib import Path
import sys
import traceback

_script_path = Path(__file__).resolve()
_roboticscripts = _script_path.parents[3]
sys.path.insert(0, str(_roboticscripts))  # imports the `simulation` namespace used by task assets
sys.path.insert(0, str(_script_path.parents[1] / 'reinforcement_learning/sac'))
from common import SOURCE_LABELS, add_launcher_options, make_env, read_checkpoint, versions


def snapshot(base, mdp, step, action):
    s = mdp.state(base)
    peg = mdp.local_peg(base)[0]
    q = base.scene['object'].data.root_quat_w[0]
    row = dict(step=step, time_s=step * base.step_dt, action=action[0].tolist(),
               peg_root_m=peg.tolist(), peg_quat_wxyz=q.tolist(),
               grasp_aligned=bool(q[1:].norm() < 0.005), holder=int(s.holder[0]),
               candidate1_substeps=int(s.candidate1[0]),
               candidate2_substeps=int(s.candidate2[0]),
               metrics={k: bool(getattr(s, k)[0]) for k in mdp.METRICS})
    for arm, offset in ((1, mdp.g.GRASP1), (2, mdp.g.GRASP2)):
        tip = mdp.ee(base, arm)[0] - base.scene.env_origins[0]
        distance = float((tip - peg - peg.new_tensor(offset)).norm())
        measured_jaw = float(mdp.jaw(base, arm)[0])
        row[f'psm{arm}'] = dict(tip_m=tip.tolist(), grasp_distance_mm=distance * 1000,
                               jaw_mean_rad=measured_jaw, jaw_closed=measured_jaw < 0.18,
                               in_capture_zone=distance < mdp.g.GRASP_RADIUS)
    return row


def summarize(rows):
    result = {}
    for arm in (1, 2):
        name = f'psm{arm}'
        if not rows:
            result[name] = None
            continue
        closest = min(rows, key=lambda r: r[name]['grasp_distance_mm'])
        result[name] = dict(min_grasp_distance_mm=closest[name]['grasp_distance_mm'],
                            closest_step=closest['step'],
                            closest_jaw_mean_rad=closest[name]['jaw_mean_rad'],
                            closest_jaw_action=closest['action'][6 if arm == 1 else 13],
                            capture_samples=sum(r[name]['in_capture_zone'] for r in rows),
                            closed_samples=sum(r[name]['jaw_closed'] for r in rows),
                            capture_closed_aligned_samples=sum(
                                r[name]['in_capture_zone'] and r[name]['jaw_closed']
                                and r['grasp_aligned'] for r in rows))
    return result


def mask_lift_unused_actions(action):
    """Diagnostic ablation: PSM1 translation/jaw only, receiver stationary/open."""
    masked = action.copy()
    masked[:, 3:6] = 0
    masked[:, 7:13] = 0
    masked[:, 13] = 1
    return masked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--phase', choices=('lift', 'handover', 'full'), default='lift')
    parser.add_argument('--source', choices=('random', 'near', *SOURCE_LABELS), default='L5')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--max_steps', type=int, default=800)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mask_lift_unused_actions', action='store_true',
                        help='Diagnostic only: freeze PSM1 rotation and PSM2; keep learned PSM1 translation/jaw')
    add_launcher_options(parser)
    args = parser.parse_args()
    args.num_envs = 1
    if args.max_steps <= 0:
        parser.error('--max_steps must be positive')
    if args.mask_lift_unused_actions and args.phase != 'lift':
        parser.error('--mask_lift_unused_actions is only valid for lift')
    trace = args.output.with_suffix('.trace.jsonl')
    if args.output.exists() or trace.exists():
        parser.error('Output already exists; choose a new filename to preserve it')
    meta = read_checkpoint(args.checkpoint)  # provenance validation before AppLauncher
    from isaaclab.app import AppLauncher
    app, env, rows, terminal = None, None, [], None
    interrupted = False
    try:
        print('[POLICY PROBE] Starting Isaac Sim', flush=True)
        app = AppLauncher(args).app
        print('[POLICY PROBE] Isaac Sim ready; importing SAC and creating the environment', flush=True)
        from stable_baselines3 import SAC
        print('[POLICY PROBE] Calling make_env', flush=True)
        env, _ = make_env(args, args.phase)
        print('[POLICY PROBE] Environment ready; importing task telemetry', flush=True)
        # make_env first imports robotic.surgery.tasks to register the Gym task
        # and resolve the extension packages. Import its MDP only afterwards.
        from robotic.surgery.tasks.surgical.peg_transfer import sac_mdp as mdp
        print('[POLICY PROBE] Loading checkpoint and resetting environment', flush=True)
        model = SAC.load(args.checkpoint, env=env, device=args.device)
        obs = env.reset()
        print('[POLICY PROBE] Collecting one policy episode', flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with trace.open('x') as file:
                for step in range(args.max_steps):
                    if not app.is_running():
                        interrupted = True
                        break
                    action, _ = model.predict(obs, deterministic=True)
                    raw_action = action.copy()
                    if args.mask_lift_unused_actions:
                        action = mask_lift_unused_actions(action)
                    row = snapshot(env.unwrapped, mdp, step, action)
                    row['raw_policy_action'] = raw_action[0].tolist()
                    rows.append(row)
                    file.write(json.dumps(row) + '\n')
                    obs, _, _, infos = env.step(action)
                    if 'transfer' in infos[0]:
                        terminal = dict(**infos[0]['transfer'], **infos[0]['episode'],
                                        timeout=infos[0]['TimeLimit.truncated'])
                        break  # never record automatically reset state as terminal state
        except KeyboardInterrupt:
            interrupted = True
        report = dict(checkpoint=str(args.checkpoint.resolve()), fingerprint=meta['fingerprint'],
                      trained_phase=meta['phase'], evaluated_phase=args.phase,
                      source=args.source, seed=args.seed, versions=versions(),
                      num_envs=1, deterministic=True, samples=len(rows),
                      action_mask='psm1_translation_jaw_only' if args.mask_lift_unused_actions else None,
                      sample_dt_s=env.unwrapped.step_dt, interrupted=interrupted,
                      completed_episode=terminal is not None, terminal=terminal,
                      diagnostics=summarize(rows), trace=str(trace),
                      note='Pre-action samples at control frequency; not all physics substeps.')
        with args.output.open('x') as file:
            file.write(json.dumps(report, indent=2) + '\n')
        print('[POLICY PROBE]', json.dumps(report, indent=2), flush=True)
    except Exception:
        print('[POLICY PROBE] Diagnostic failed; full exception follows', file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == '__main__':
    main()
