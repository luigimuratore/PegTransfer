"""Deterministic episode metrics from pre-reset evidence, independent of rewards."""

import json

from common import make_env, parser, read_checkpoint, versions


def wilson(count, n):
    if not n:
        return None
    z, p = 1.96, count / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return [max(0, center - half), min(1, center + half)]


def main():
    args = parser('evaluate')
    meta = read_checkpoint(args.checkpoint)
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    env, rows = None, []
    interrupted = False
    try:
        from stable_baselines3 import SAC
        env, _ = make_env(args, args.phase)
        model = SAC.load(args.checkpoint, env=env, device=args.device)
        obs = env.reset()
        try:
            while len(rows) < args.episodes and app.is_running():
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, infos = env.step(action)
                for i, info in enumerate(infos):
                    if 'transfer' in info and len(rows) < args.episodes:
                        rows.append(dict(env_id=i, **info['transfer'], **info['episode'],
                                         timeout=info['TimeLimit.truncated']))
                if any('transfer' in info for info in infos):
                    print(f'[EVAL] {len(rows)}/{args.episodes}', flush=True)
        except KeyboardInterrupt:
            interrupted = True
        from robotic.surgery.tasks.surgical.peg_transfer.sac_mdp import METRICS, g
        result = dict(checkpoint=str(args.checkpoint.resolve()), trained_phase=meta['phase'],
                      evaluated_phase=args.phase, seed=args.seed, versions=versions(), fingerprint=meta['fingerprint'],
                      episodes=len(rows), requested_episodes=args.episodes, interrupted=interrupted,
                      complete=len(rows) == args.episodes, source=args.source, num_envs=args.num_envs,
                      assisted_grasp=True)
        for name in METRICS:
            n = sum(r['metrics'][name] for r in rows)
            result[name] = n
            result[name + '_rate'] = n / len(rows) if rows else None
            result[name + '_ci95'] = wilson(n, len(rows))
        result['timeouts'] = sum(r['timeout'] for r in rows)
        result['failures'] = sum(r['failure'] for r in rows)
        result['blocked_moves'] = sum(r['blocked_moves'] for r in rows)
        result['by_source'] = {}
        result['source_positions'] = {name: list(map(float, xy)) for name, xy in zip(g.SOURCE_LABELS, g.SOURCE_POSTS)}
        for source in g.SOURCE_LABELS:
            subset = [r for r in rows if r['source'] == source]
            result['by_source'][source] = dict(episodes=len(subset), **{
                k + '_rate': sum(r['metrics'][k] for r in subset) / len(subset) if subset else None for k in METRICS})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        args.output.with_suffix('.episodes.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
