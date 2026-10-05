# SPDX-License-Identifier: BSD-3-Clause
"""Measure physical peg-transfer milestones on complete evaluation episodes."""

import argparse
import json
import os
import re

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from checkpoint_utils import resolve_run_name  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a dual-PSM peg-transfer policy.")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--task", default="Isaac-Peg-Transfer-Dual-PSM-Play-v0")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.episodes < 1 or args.num_envs < 1:
    parser.error("--episodes and --num_envs must be positive")
app = AppLauncher(args).app

import gymnasium as gym
import robotic.surgery.tasks  # noqa: F401
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    agent_cfg = cli_args.parse_rsl_rl_cfg(args.task, args)
    root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args.load_run is not None:
        agent_cfg.load_run = re.escape(resolve_run_name(root, args.load_run)) + "$"
    checkpoint = get_checkpoint_path(root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = env.get_observations()

    manager = env.unwrapped.reward_manager
    event_indices = {name: manager.active_terms.index(name) for name in
                     ("first_grasp_event", "lift_event", "stable_lift_event", "receiver_grasp_event", "handover_event")}
    seen = {name: torch.zeros(env.num_envs, dtype=torch.bool, device=env.unwrapped.device)
            for name in event_indices}
    results = {"episodes": 0, "first_grasped": 0, "lifted": 0, "stable_lift": 0, "receiver_grasped": 0, "handover": 0,
               "full_success": 0, "timeouts": 0, "dropped": 0}
    steps = 0
    interrupted = False
    try:
        while results["episodes"] < args.episodes:
            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))
            steps += 1
            for name, index in event_indices.items():
                seen[name] |= manager._step_reward[:, index] > 0
            base = env.unwrapped
            done = base.reset_buf
            if done.any():
                remaining = args.episodes - results["episodes"]
                ids = done.nonzero(as_tuple=False).squeeze(-1)[:remaining]
                results["episodes"] += ids.numel()
                results["first_grasped"] += int(seen["first_grasp_event"][ids].sum())
                results["lifted"] += int(seen["lift_event"][ids].sum())
                results["stable_lift"] += int(seen["stable_lift_event"][ids].sum())
                results["receiver_grasped"] += int(seen["receiver_grasp_event"][ids].sum())
                results["handover"] += int(seen["handover_event"][ids].sum())
                results["full_success"] += int(base.termination_manager.get_term("success")[ids].sum())
                results["timeouts"] += int(base.reset_time_outs[ids].sum())
                results["dropped"] += int(base.termination_manager.get_term("object_dropping")[ids].sum())
                for values in seen.values():
                    values[done] = False
                print(f"[EVAL] {results['episodes']}/{args.episodes} episodes completed", flush=True)
    except KeyboardInterrupt:
        interrupted = True
    env.close()
    results["steps"] = steps
    results["interrupted"] = interrupted
    results["checkpoint"] = checkpoint
    for key in ("first_grasped", "lifted", "stable_lift", "receiver_grasped", "handover", "full_success"):
        results[key + "_rate"] = results[key] / results["episodes"] if results["episodes"] else None
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
