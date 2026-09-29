"""Open the dual-PSM peg-transfer scene without requiring a trained policy."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import robotic.surgery.tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main():
    cfg = parse_env_cfg("Isaac-Peg-Transfer-Dual-PSM-Play-v0", num_envs=args.num_envs)
    env = gym.make("Isaac-Peg-Transfer-Dual-PSM-Play-v0", cfg=cfg)
    env.reset()
    print("Peg-transfer scene ready: two PSMs, board, source peg on L5/L6, target R2.")
    try:
        # Let Kit render the scene. No policy action is applied in this preview.
        while simulation_app.is_running():
            simulation_app.update()
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
