# SPDX-License-Identifier: BSD-3-Clause

"""Assisted-grasp observations, staged rewards and verified peg transfer."""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.utils.math import subtract_frame_transforms

from .scene_cfg import PEG_OFFSET_X, PEG_OFFSET_Y, PEG_Z, POST_CLEARANCE_ROOT_Z

# Root poses for the source and destination are defined by the same offset from a post.
TARGET_POST_LOCAL = (0.042 + PEG_OFFSET_X, 0.015 + PEG_OFFSET_Y, PEG_Z)
HANDOVER_LOCAL = (0.020, -0.0075, 0.065)
RECEIVER_READY_LOCAL = (0.040, -0.0075, 0.065)
ABOVE_TARGET_LOCAL = (TARGET_POST_LOCAL[0], TARGET_POST_LOCAL[1], 0.060)
REST_HEIGHT = 0.025  # USD block root height when resting on the board.
LIFT_HEIGHT = 0.055
# Offset from the USD root to the reachable side of the block.
PSM1_GRASP_OFFSET = (0.003, -0.009, -0.006)
GRASP_RADIUS = 0.012
GRASP_DWELL_STEPS = 2
STABLE_LIFT_STEPS = 15  # 0.3 s at the 50 Hz policy rate.


def _world_point(env, point: tuple[float, float, float]) -> torch.Tensor:
    return env.scene.env_origins + env.scene.env_origins.new_tensor(point)


def _object_w(env, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    return env.scene[object_cfg.name].data.root_pos_w[:, :3]


def _ee_w(env, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[ee_frame_cfg.name].data.target_pos_w[:, 0, :]


def _distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(a - b, dim=1)


def _proximity(distance: torch.Tensor, std: float) -> torch.Tensor:
    return torch.exp(-distance / std)


def _gripper_joint_position(env, arm: int) -> torch.Tensor:
    robot = env.scene[f"robot_{arm}"]
    if not hasattr(env, "_peg_gripper_joint_ids"):
        env._peg_gripper_joint_ids = {
            index: env.scene[f"robot_{index}"].find_joints("psm_tool_gripper.*_joint")[0]
            for index in (1, 2)
        }
    return robot.data.joint_pos[:, env._peg_gripper_joint_ids[arm]].abs().mean(dim=1)


def _closed_score(env, arm: int) -> torch.Tensor:
    return ((0.34 - _gripper_joint_position(env, arm)) / 0.25).clamp(0.0, 1.0)


def _open_gripper(env, arm: int) -> torch.Tensor:
    return _gripper_joint_position(env, arm) > 0.38


def _open_fraction(env, arm: int) -> torch.Tensor:
    return ((_gripper_joint_position(env, arm) - 0.10) / 0.30).clamp(0.0, 1.0)


def grip_score(env, arm: int) -> torch.Tensor:
    """Observed assisted hold, including a receiver already latched during handover."""
    state = _state(env)
    held = state.holder == arm
    if arm == 2:
        held |= state.receiver_latched
    return held.float()


def _gripping(env, arm: int) -> torch.Tensor:
    return grip_score(env, arm) > 0.5


def _lift_fraction(env) -> torch.Tensor:
    return ((_object_w(env)[:, 2] - REST_HEIGHT) / (LIFT_HEIGHT - REST_HEIGHT)).clamp(0.0, 1.0)


class TransferState:
    """Per-environment assisted-hold state and evidence of transfer sequence."""

    def __init__(self, env):
        n = env.num_envs
        device = env.device
        self.psm1_hold_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.holder = torch.zeros(n, dtype=torch.long, device=device)
        self.receiver_latched = torch.zeros(n, dtype=torch.bool, device=device)
        self.hold_offset = torch.zeros(n, 3, device=device)
        self.source_peg_xy = torch.zeros(n, 2, device=device)
        self.source_cleared = torch.zeros(n, dtype=torch.bool, device=device)
        self.psm1_candidate_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.psm2_candidate_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.psm2_hold_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.first_grasped = torch.zeros(n, dtype=torch.bool, device=device)
        self.lifted = torch.zeros(n, dtype=torch.bool, device=device)
        self.stable_lift_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.stable_lift_achieved = torch.zeros(n, dtype=torch.bool, device=device)
        self.new_stable_lift = torch.zeros(n, dtype=torch.bool, device=device)
        self.previous_peg_pos = torch.zeros(n, 3, device=device)
        self.previous_peg_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.receiver_grasped = torch.zeros(n, dtype=torch.bool, device=device)
        self.handover = torch.zeros(n, dtype=torch.bool, device=device)
        self.settled_steps = torch.zeros(n, dtype=torch.long, device=device)
        self.new_first_grasp = torch.zeros(n, dtype=torch.bool, device=device)
        self.new_lift = torch.zeros(n, dtype=torch.bool, device=device)
        self.new_receiver_grasp = torch.zeros(n, dtype=torch.bool, device=device)
        self.new_handover = torch.zeros(n, dtype=torch.bool, device=device)
        self.last_step = -1

    def reset(self, env_ids):
        for name in (
            "psm1_hold_steps", "psm2_hold_steps", "holder", "receiver_latched",
            "hold_offset", "source_peg_xy", "source_cleared", "psm1_candidate_steps", "psm2_candidate_steps",
            "first_grasped", "lifted", "stable_lift_steps", "stable_lift_achieved",
            "new_stable_lift", "previous_peg_pos", "previous_peg_valid", "receiver_grasped",
            "handover", "settled_steps", "new_first_grasp", "new_lift", "new_receiver_grasp", "new_handover",
        ):
            getattr(self, name)[env_ids] = 0


def _state(env) -> TransferState:
    if not hasattr(env, "_peg_transfer_state"):
        env._peg_transfer_state = TransferState(env)
    return env._peg_transfer_state


def update_assisted_grasp(env, env_ids=None):
    """Latch a nearby closed gripper and translate the peg with its tool tip.

    The peg follows a nearby closed gripper until release. While still threaded
    on the source post, it may only move vertically; the post and board have
    physical colliders for the dynamic peg.
    """
    state = _state(env)
    obj = env.scene["object"]
    peg = obj.data.root_pos_w[:, :3]
    ee1 = _ee_w(env, SceneEntityCfg("ee_1_frame"))
    ee2 = _ee_w(env, SceneEntityCfg("ee_2_frame"))
    closed1 = _closed_score(env, 1) > 0.5
    closed2 = _closed_score(env, 2) > 0.5

    near1 = _distance(ee1, peg + peg.new_tensor(PSM1_GRASP_OFFSET)) < GRASP_RADIUS
    candidate1 = (state.holder == 0) & closed1 & near1
    state.psm1_candidate_steps = torch.where(candidate1, state.psm1_candidate_steps + 1, 0)
    attach1 = candidate1 & (state.psm1_candidate_steps >= GRASP_DWELL_STEPS)
    state.holder[attach1] = 1
    state.hold_offset[attach1] = peg[attach1] - ee1[attach1]
    state.source_peg_xy[attach1] = peg[attach1, :2]
    state.source_cleared[attach1] = peg[attach1, 2] >= (
        env.scene.env_origins[attach1, 2] + POST_CLEARANCE_ROOT_Z
    )

    receiver_point = peg + peg.new_tensor((0.012, 0.0, 0.0))
    near2 = _distance(ee2, receiver_point) < GRASP_RADIUS
    # The receiver must not take the peg while it is still threaded on L5/L6.
    candidate2 = (state.holder == 1) & state.source_cleared & closed2 & near2
    state.psm2_candidate_steps = torch.where(candidate2, state.psm2_candidate_steps + 1, 0)
    state.receiver_latched |= candidate2 & (state.psm2_candidate_steps >= GRASP_DWELL_STEPS)
    state.receiver_latched &= closed2

    release1 = (state.holder == 1) & _open_gripper(env, 1)
    transfer = release1 & state.receiver_latched
    state.holder[release1] = 0
    state.holder[transfer] = 2
    state.hold_offset[transfer] = peg[transfer] - ee2[transfer]
    state.receiver_latched[release1] = False
    release2 = (state.holder == 2) & _open_gripper(env, 2)
    state.holder[release2] = 0

    held_ids = (state.holder > 0).nonzero(as_tuple=False).squeeze(-1)
    if held_ids.numel() == 0:
        return
    held_by_1 = state.holder[held_ids] == 1
    tool_pos = torch.where(held_by_1[:, None], ee1[held_ids], ee2[held_ids])
    root_pose = obj.data.root_pose_w[held_ids].clone()
    root_pose[:, :3] = tool_pos + state.hold_offset[held_ids]
    root_pose[:, 2] = torch.maximum(root_pose[:, 2], env.scene.env_origins[held_ids, 2] + PEG_Z)
    before_clear = held_by_1 & (~state.source_cleared[held_ids])
    clearance_z = env.scene.env_origins[held_ids, 2] + POST_CLEARANCE_ROOT_Z
    lateral_error = torch.linalg.vector_norm(root_pose[:, :2] - state.source_peg_xy[held_ids], dim=1)
    # A tool that pulls away sideways before lifting clear loses the assisted hold.
    lost = before_clear & (root_pose[:, 2] < clearance_z) & (lateral_error > GRASP_RADIUS)
    state.holder[held_ids[lost]] = 0
    state.receiver_latched[held_ids[lost]] = False
    keep = ~lost
    held_ids, held_by_1, tool_pos, root_pose = (
        held_ids[keep], held_by_1[keep], tool_pos[keep], root_pose[keep]
    )
    if held_ids.numel() == 0:
        return
    before_clear = held_by_1 & (~state.source_cleared[held_ids])
    # Prevent teleporting the block sideways through the source cylinder.
    root_pose[before_clear, :2] = state.source_peg_xy[held_ids[before_clear]]
    just_cleared = before_clear & (
        root_pose[:, 2] >= env.scene.env_origins[held_ids, 2] + POST_CLEARANCE_ROOT_Z
    )
    state.source_cleared[held_ids[just_cleared]] = True
    # Keep the next update continuous after the vertical-only part of the lift.
    state.hold_offset[held_ids[just_cleared], :2] = (
        root_pose[just_cleared, :2] - tool_pos[just_cleared, :2]
    )
    root_pose[:, 3:7] = root_pose.new_tensor((1.0, 0.0, 0.0, 0.0))
    obj.write_root_pose_to_sim(root_pose, env_ids=held_ids)
    obj.write_root_velocity_to_sim(torch.zeros(held_ids.numel(), 6, device=env.device), env_ids=held_ids)


def reset_transfer_state(env, env_ids: torch.Tensor):
    """Clear phase history after each episode, including asynchronous vector resets."""
    _state(env).reset(env_ids)


def update_transfer_state(env) -> TransferState:
    state = _state(env)
    if state.last_step == env.common_step_counter:
        return state
    state.last_step = env.common_step_counter
    p1_grip = _gripping(env, 1)
    p2_grip = _gripping(env, 2)
    state.psm1_hold_steps = torch.where(p1_grip, state.psm1_hold_steps + 1, 0)
    state.psm2_hold_steps = torch.where(p2_grip, state.psm2_hold_steps + 1, 0)
    height = _object_w(env)[:, 2]

    state.new_first_grasp = (~state.first_grasped) & (state.psm1_hold_steps >= 1)
    state.first_grasped |= state.new_first_grasp
    state.new_lift = (~state.lifted) & (state.psm1_hold_steps >= 1) & (height > 0.045)
    state.lifted |= state.new_lift
    peg = env.scene["object"].data.root_pos_w[:, :3]
    peg_speed = torch.linalg.vector_norm(peg - state.previous_peg_pos, dim=1) / env.step_dt
    steady_lift = (
        state.previous_peg_valid & p1_grip & (~_open_gripper(env, 1))
        & (height >= 0.055) & (height <= 0.080) & (peg_speed < 0.08)
    )
    state.stable_lift_steps = torch.where(steady_lift, state.stable_lift_steps + 1, 0)
    state.new_stable_lift = (~state.stable_lift_achieved) & (state.stable_lift_steps >= STABLE_LIFT_STEPS)
    state.stable_lift_achieved |= state.new_stable_lift
    state.previous_peg_pos[:] = peg
    state.previous_peg_valid[:] = True
    state.new_receiver_grasp = (
        state.stable_lift_achieved & (~state.receiver_grasped) & (state.psm2_hold_steps >= 1) & (height > 0.045)
    )
    state.receiver_grasped |= state.new_receiver_grasp
    state.new_handover = (
        state.receiver_grasped & (~state.handover) & p2_grip & (~p1_grip)
        & _open_gripper(env, 1) & (height > 0.045)
    )
    state.handover |= state.new_handover

    delta = peg - _world_point(env, TARGET_POST_LOCAL)
    xy_ok = torch.linalg.vector_norm(delta[:, :2], dim=1) < 0.008
    z_ok = delta[:, 2].abs() < 0.010
    stable = torch.linalg.vector_norm(env.scene["object"].data.root_lin_vel_w, dim=1) < 0.025
    clear_1 = _distance(_ee_w(env, SceneEntityCfg("ee_1_frame")), peg) > 0.025
    clear_2 = _distance(_ee_w(env, SceneEntityCfg("ee_2_frame")), peg) > 0.025
    placed = state.handover & xy_ok & z_ok & stable & clear_1 & clear_2 & _open_gripper(env, 2)
    state.settled_steps = torch.where(placed, state.settled_steps + 1, 0)
    return state


def peg_transfer_success(env) -> torch.Tensor:
    """R2 placement for five steps after assisted lift, receipt and giver release."""
    return update_transfer_state(env).settled_steps >= 5


def first_lift_completed(env) -> torch.Tensor:
    """Finish lift training only after the peg is held nearly still for 0.3 s."""
    return update_transfer_state(env).stable_lift_achieved


def handover_completed(env) -> torch.Tensor:
    return update_transfer_state(env).handover


def phase_observation(env) -> torch.Tensor:
    """Markov-relevant task progress and measured contact for the policy."""
    state = _state(env)
    return torch.stack(
        (
            _lift_fraction(env), grip_score(env, 1), grip_score(env, 2),
            state.lifted.float(), state.receiver_grasped.float(), state.handover.float(),
        ),
        dim=1,
    )


def object_velocity(env, object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    return env.scene[object_cfg.name].data.root_lin_vel_w


def object_to_target(env) -> torch.Tensor:
    return _world_point(env, TARGET_POST_LOCAL) - _object_w(env)


def ee_to_object(env, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    return _object_w(env) - _ee_w(env, ee_frame_cfg)


def ee_to_psm1_grasp(env) -> torch.Tensor:
    peg = _object_w(env)
    return peg + peg.new_tensor(PSM1_GRASP_OFFSET) - _ee_w(env, SceneEntityCfg("ee_1_frame"))


def _psm1_grasp_distance(env) -> torch.Tensor:
    return torch.linalg.vector_norm(ee_to_psm1_grasp(env), dim=1)


def object_position_in_robot_root_frame(
    env, robot_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    position, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], _object_w(env, object_cfg)
    )
    return position


def _progress_measure(env, kind: str) -> torch.Tensor:
    peg = _object_w(env)
    if kind == "psm1_reach":
        return _proximity(_psm1_grasp_distance(env), 0.035)
    if kind == "psm1_precision":
        return _proximity(_psm1_grasp_distance(env), 0.007)
    if kind == "psm1_open_approach":
        return _proximity(_psm1_grasp_distance(env), 0.025) * _open_fraction(env, 1)
    if kind == "psm1_lift":
        return _lift_fraction(env) * grip_score(env, 1)
    if kind == "handover_position":
        return (
            _proximity(_distance(peg, _world_point(env, HANDOVER_LOCAL)), 0.055)
            * _lift_fraction(env) * grip_score(env, 1)
        )
    if kind == "psm2_reach":
        # The receiving tool approaches the opposite face of the lifted block.
        receiver_point = peg + peg.new_tensor((0.012, 0.0, 0.0))
        return _proximity(_distance(_ee_w(env, SceneEntityCfg("ee_2_frame")), receiver_point), 0.035) * _lift_fraction(env)
    if kind == "transport":
        return (
            _proximity(_distance(peg, _world_point(env, ABOVE_TARGET_LOCAL)), 0.055)
            * _state(env).handover.float()
        )
    if kind == "placement":
        return (
            _proximity(_distance(peg, _world_point(env, TARGET_POST_LOCAL)), 0.030)
            * _state(env).handover.float()
        )
    raise ValueError(f"Unknown peg-transfer progress measure: {kind}")


class ProgressReward(ManagerTermBase):
    """Reward change in task potential, avoiding payment for waiting near an object.

    Isaac Lab multiplies each reward term by step_dt. Dividing the difference by
    step_dt makes the configured weight the value of a completed transition.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous = torch.zeros(env.num_envs, device=env.device)
        self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(self, env, kind: str) -> torch.Tensor:
        current = _progress_measure(env, kind)
        if kind == "psm1_open_approach":
            # Remember the best open approach: closing for a real grasp must not erase its bonus.
            current = torch.maximum(current, self.previous)
        change = torch.where(self.initialized, current - self.previous, 0.0)
        self.previous = current.clone()
        self.initialized[:] = True
        return change / env.step_dt

    def reset(self, env_ids=None):
        self.previous[env_ids] = 0.0
        self.initialized[env_ids] = False


def psm1_grasp_reward(env) -> torch.Tensor:
    return grip_score(env, 1)


def psm2_ready_reward(env) -> torch.Tensor:
    distance = _distance(_ee_w(env, SceneEntityCfg("ee_2_frame")), _world_point(env, RECEIVER_READY_LOCAL))
    return _proximity(distance, 0.05) * (1.0 - _lift_fraction(env))


def psm2_grasp_reward(env) -> torch.Tensor:
    return grip_score(env, 2) * _lift_fraction(env)


def first_grasp_milestone(env) -> torch.Tensor:
    return update_transfer_state(env).new_first_grasp.float()


def lift_milestone(env) -> torch.Tensor:
    return update_transfer_state(env).new_lift.float()


def stable_hold_reward(env) -> torch.Tensor:
    state = update_transfer_state(env)
    return torch.where(
        state.stable_lift_achieved,
        torch.zeros_like(state.stable_lift_steps, dtype=torch.float),
        state.stable_lift_steps.clamp(max=STABLE_LIFT_STEPS).float() / STABLE_LIFT_STEPS,
    )


def stable_lift_milestone(env) -> torch.Tensor:
    return update_transfer_state(env).new_stable_lift.float()


def receiver_grasp_milestone(env) -> torch.Tensor:
    return update_transfer_state(env).new_receiver_grasp.float()


def handover_milestone(env) -> torch.Tensor:
    return update_transfer_state(env).new_handover.float()


def elapsed_time(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)
