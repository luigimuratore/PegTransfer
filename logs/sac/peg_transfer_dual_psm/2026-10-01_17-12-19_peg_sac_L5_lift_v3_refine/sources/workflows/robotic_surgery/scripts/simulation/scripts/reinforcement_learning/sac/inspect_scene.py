"""User-run Isaac Sim geometry smoke check, without a learned policy.

Checks L5/L6/R2 settling, off-centre post collision, and absence of false success.
This script intentionally runs PhysX; do not include it in static CI.
"""

import argparse
import json
from pathlib import Path


def main():
    from common import add_launcher_options
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num_envs', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=Path, default=Path('logs/sac/scene_check.json'))
    add_launcher_options(p)
    args = p.parse_args()
    if args.num_envs != 1:
        p.error('Run geometry inspection with exactly one environment')
    args.source, args.video = 'random', False
    from common import make_env
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    env = None
    try:
        import torch
        from pxr import UsdPhysics
        from isaacsim.core.utils.stage import get_current_stage
        from robotic.surgery.tasks.surgical.peg_transfer import sac_geometry as g, sac_mdp as mdp
        env, _ = make_env(args, 'full')
        base = env.unwrapped
        env.reset()
        actions = torch.zeros(1, 14, device=base.device)
        actions[:, [6, 13]] = 1
        results = []
        for name, xy, offset in [('L5', g.SOURCE_POSTS[0], 0.0), ('L6', g.SOURCE_POSTS[1], 0.0),
                                  ('R2', (0.042, 0.015), 0.0), ('R2_wall', (0.042, 0.015), 0.004)]:
            env.reset()
            pose = base.scene['object'].data.root_pose_w.clone()
            pose[:, :3] = pose.new_tensor((xy[0] - g.HOLE_CENTER[0] + offset, xy[1],
                                           g.CLEAR_Z + 0.004 if offset else g.RESET_Z)) + base.scene.env_origins
            pose[:, 3:] = pose.new_tensor((1, 0, 0, 0))
            base.scene['object'].write_root_pose_to_sim(pose)
            base.scene['object'].write_root_velocity_to_sim(torch.zeros(1, 6, device=base.device))
            seen_success = False
            post_contact = False
            for step in range(100):
                _, _, done, infos = env.step(actions)
                if offset and step < 15:
                    z = float(mdp.local_peg(base)[0, 2])
                    vz = float(base.scene['object'].data.root_lin_vel_w[0, 2])
                    # Contact should stop the initial fall at the post tip; later tipping is legal.
                    post_contact |= abs(z + g.PEG_MIN[2] - g.POST_TOP) < 0.002 and vz > -0.05
                seen_success |= bool(mdp.state(base).full_success[0])
                if done[0]:
                    seen_success |= infos[0]['transfer']['metrics']['full_success']
                    raise RuntimeError(f'{name}: unexpected reset during settling: {infos[0]}')
            pos = mdp.local_peg(base)[0]
            radial = float(torch.linalg.vector_norm(pos[:2] + pos.new_tensor(g.HOLE_CENTER) - pos.new_tensor(xy)))
            height = float(pos[2])
            correct = (radial < g.THREAD_TOLERANCE and abs(height - g.REST_Z) < 0.0007) if not offset else post_contact
            results.append(dict(case=name, position=pos.tolist(), upright=bool(mdp.upright(base)[0]),
                                geometry_pass=correct, false_success=seen_success))
        stage = get_current_stage()
        colliders = [str(p.GetPath()) for p in stage.Traverse()
                     if 'Object/Collider_' in str(p.GetPath()) and p.HasAPI(UsdPhysics.CollisionAPI)]
        report = dict(cases=results, compound_colliders=colliders,
                      passed=all(r['geometry_pass'] and not r['false_success'] for r in results)
                      and len(colliders) == len(g.collider_sectors()),
                      requires_visual_review=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
        if not report['passed']:
            raise RuntimeError('Geometry smoke check failed; inspect scene before training')
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
