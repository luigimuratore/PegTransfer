# SPDX-License-Identifier: BSD-3-Clause

"""Observations, rewards and success conditions for dual-PSM peg transfer."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from .scene_cfg import PEG_OFFSET_X, PEG_OFFSET_Y, PEG_Z

# The existing board uses L5/L6 as starting posts and R2 as the first target.
# Coordinates are relative to each environment origin, in metres.
TARGET_POST_LOCAL = (0.042 + PEG_OFFSET_X, 0.015 + PEG_OFFSET_Y, PEG_Z)


def _target_w(env) -> torch.Tensor:
    origins = env.scene.env_origins
    return origins + origins.new_tensor(TARGET_POST_LOCAL)


def _object_w(env, object_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[object_cfg.name].data.root_pos_w[:, :3]


def _ee_w(env, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[ee_frame_cfg.name].data.target_pos_w[:, 0, :]


def _closed_score(env, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    # The PSM gripper commands are approximately +/-0.07 closed, +/-0.5 open.
    q = env.scene[robot_cfg.name].data.joint_pos[:, robot_cfg.joint_ids].abs().mean(dim=1)
    return ((0.35 - q) / 0.23).clamp(0.0, 1.0)


def object_to_target(env, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    """Target position relative to the moving peg; independent of env origin."""
    return _target_w(env) - _object_w(env, object_cfg)


def ee_to_object(
    env,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    return _object_w(env, object_cfg) - _ee_w(env, ee_frame_cfg)


def reached_peg(
    env,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    std: float = 0.04,
) -> torch.Tensor:
    return 1.0 - torch.tanh(torch.linalg.vector_norm(ee_to_object(env, ee_frame_cfg, object_cfg), dim=1) / std)


def holding_peg(
    env,
    ee_frame_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    return reached_peg(env, ee_frame_cfg, object_cfg, std=0.025) * _closed_score(env, robot_cfg)


def peg_lifted(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    minimum_height: float = 0.05,
) -> torch.Tensor:
    return (_object_w(env, object_cfg)[:, 2] > minimum_height).float()


def receiver_reach(
    env,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    return peg_lifted(env, object_cfg) * reached_peg(env, ee_frame_cfg, object_cfg)


def receiver_holds(
    env,
    ee_frame_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    return peg_lifted(env, object_cfg) * holding_peg(env, ee_frame_cfg, robot_cfg, object_cfg)


def transport_to_target(
    env,
    ee_frame_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    std: float = 0.08,
) -> torch.Tensor:
    # Reward motion to the right post only after the receiving PSM is at the peg.
    distance = torch.linalg.vector_norm(object_to_target(env, object_cfg), dim=1)
    return receiver_holds(env, ee_frame_cfg, robot_cfg, object_cfg) * (1.0 - torch.tanh(distance / std))


def place_at_target(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    std: float = 0.015,
) -> torch.Tensor:
    # This is a diagnostic and a dense placement signal; success is stricter.
    distance = torch.linalg.vector_norm(object_to_target(env, object_cfg), dim=1)
    return 1.0 - torch.tanh(distance / std)


def peg_transfer_success(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_1_cfg: SceneEntityCfg = SceneEntityCfg("ee_1_frame"),
    ee_2_cfg: SceneEntityCfg = SceneEntityCfg("ee_2_frame"),
    robot_2_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot_2", joint_names=["psm_tool_gripper.*_joint"]
    ),
) -> torch.Tensor:
    """Peg settled over R2, released by PSM2, with both tools clear."""
    obj = env.scene[object_cfg.name]
    peg_w = obj.data.root_pos_w[:, :3]
    delta = peg_w - _target_w(env)
    xy_ok = torch.linalg.vector_norm(delta[:, :2], dim=1) < 0.008
    z_ok = delta[:, 2].abs() < 0.012
    stable = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=1) < 0.03
    psm_1_clear = torch.linalg.vector_norm(_ee_w(env, ee_1_cfg) - peg_w, dim=1) > 0.025
    psm_2_clear = torch.linalg.vector_norm(_ee_w(env, ee_2_cfg) - peg_w, dim=1) > 0.025
    psm_2_open = _closed_score(env, robot_2_cfg) < 0.2
    return xy_ok & z_ok & stable & psm_1_clear & psm_2_clear & psm_2_open


def object_position_in_robot_root_frame(
    env,
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Peg position expressed in the selected PSM base frame."""
    robot = env.scene[robot_cfg.name]
    object_pos_w = _object_w(env, object_cfg)
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b
