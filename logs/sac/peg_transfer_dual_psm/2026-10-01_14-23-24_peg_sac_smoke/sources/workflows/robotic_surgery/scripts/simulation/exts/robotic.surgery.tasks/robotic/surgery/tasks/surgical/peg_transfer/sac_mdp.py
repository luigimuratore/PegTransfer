"""SAC task state, guarded translational assistance and ordered episode evidence.

No Isaac imports: the sequence and geometry are exercised by CPU regression tests.
Attachment is an approximation, not a contact/force based grasp.
"""

import torch

from . import sac_geometry as g

METRICS = ('grasp_psm1', 'lift_stable', 'grasp_psm2', 'handover', 'placement', 'full_success')


def ee(env, arm):
    return env.scene[f'ee_{arm}_frame'].data.target_pos_w[:, 0]


def jaw(env, arm):
    if not hasattr(env, '_sac_jaw_ids'):
        env._sac_jaw_ids = {a: env.scene[f'robot_{a}'].find_joints('psm_tool_gripper.*_joint')[0]
                            for a in (1, 2)}
    return env.scene[f'robot_{arm}'].data.joint_pos[:, env._sac_jaw_ids[arm]].abs().mean(-1)


def local_peg(env):
    return env.scene['object'].data.root_pos_w - env.scene.env_origins


class State:
    def __init__(self, env):
        n, d = env.num_envs, env.device
        for name in ('receiver_latched', 'source_cleared', 'previous_valid', *METRICS):
            setattr(self, name, torch.zeros(n, dtype=torch.bool, device=d))
        for name in ('holder', 'candidate1', 'candidate2', 'stable_steps', 'handover_steps', 'settled_steps'):
            setattr(self, name, torch.zeros(n, dtype=torch.long, device=d))
        for name, width in (('offset', 3), ('previous', 3), ('source', 3), ('highwater', 6)):
            setattr(self, name, torch.zeros(n, width, device=d))
        self.events = torch.zeros(n, 6, device=d)
        self.blocked_moves = torch.zeros(n, dtype=torch.long, device=d)
        self.last_step = -1

    def reset(self, ids):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                value[ids] = 0


def state(env):
    if not hasattr(env, '_sac_transfer'):
        env._sac_transfer = State(env)
    return env._sac_transfer


def reset(env, env_ids=None):
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    s = state(env)
    s.reset(env_ids)
    forced = getattr(env.cfg, 'source_post', 'random')
    choice = torch.randint(2, (len(env_ids),), device=env.device)
    if forced != 'random':
        choice[:] = 0 if forced == 'L5' else 1
    xy = env.scene.env_origins.new_tensor(g.SOURCE_POSTS)[choice]
    pose = env.scene['object'].data.default_root_state[env_ids, :7].clone()
    pose[:, :2] = xy - pose.new_tensor(g.HOLE_CENTER)
    pose[:, 2] = g.RESET_Z
    s.source[env_ids] = pose[:, :3]
    pose[:, :3] += env.scene.env_origins[env_ids]
    pose[:, 3:] = pose.new_tensor((1, 0, 0, 0))
    env.scene['object'].write_root_pose_to_sim(pose, env_ids=env_ids)
    env.scene['object'].write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)


def upright(env):
    # This also rejects yaw: the assisted collider and fixed grasp offsets use identity orientation.
    q = env.scene['object'].data.root_quat_w
    return torch.linalg.vector_norm(q[:, 1:], dim=1) < 0.04


def hole_on_target(env):
    """Both ends of the tilted hole must contain the R2 axis with clearance."""
    p, q = local_peg(env), env.scene['object'].data.root_quat_w
    endpoints = p.new_tensor([(*g.HOLE_CENTER, g.PEG_MIN[2]), (*g.HOLE_CENTER, g.PEG_MAX[2])])
    v = endpoints[None].expand(env.num_envs, -1, -1)
    xyz = q[:, None, 1:].expand_as(v)
    uv = torch.linalg.cross(xyz, v, dim=-1)
    rotated = v + 2 * (q[:, None, :1] * uv + torch.linalg.cross(xyz, uv, dim=-1))
    radial = torch.linalg.vector_norm(p[:, None, :2] + rotated[:, :, :2] - p.new_tensor((0.042, 0.015)), dim=-1)
    return (radial < g.THREAD_TOLERANCE).all(dim=1)


def assist(env):
    """Called after each physics substep, before rewards, terminations and observations.

    Fixed identity orientation; no grasp forces, compliance or torque transmission.
    Every proposed translation is speed-limited and swept-tested. A rejected move
    leaves the peg where it is and eventually breaks a stretched attachment.
    """
    s = state(env)
    obj = env.scene['object']
    p = local_peg(env)
    e1, e2 = ee(env, 1) - env.scene.env_origins, ee(env, 2) - env.scene.env_origins
    closed1, closed2 = jaw(env, 1) < 0.18, jaw(env, 2) < 0.18
    near1 = torch.linalg.vector_norm(e1 - p - p.new_tensor(g.GRASP1), dim=1) < g.GRASP_RADIUS
    near2 = torch.linalg.vector_norm(e2 - p - p.new_tensor(g.GRASP2), dim=1) < g.GRASP_RADIUS
    aligned = torch.linalg.vector_norm(obj.data.root_quat_w[:, 1:], dim=1) < 0.005
    candidate1 = (s.holder == 0) & ~s.handover & closed1 & near1 & aligned
    s.candidate1 = torch.where(candidate1, s.candidate1 + 1, 0)
    attach = candidate1 & (s.candidate1 >= 8)  # 40 ms at 200 Hz
    s.holder[attach] = 1
    s.offset[attach] = p[attach] - e1[attach]
    # Never latch PSM2 before evidence of a stable donor lift.
    candidate2 = (s.holder == 1) & s.lift_stable & s.source_cleared & closed2 & near2
    s.candidate2 = torch.where(candidate2, s.candidate2 + 1, 0)
    s.receiver_latched |= candidate2 & (s.candidate2 >= 8)
    s.receiver_latched &= closed2 & near2 & (s.holder == 1)
    release1 = (s.holder == 1) & (jaw(env, 1) > 0.38)
    transfer = release1 & s.receiver_latched & s.grasp_psm2
    s.holder[release1] = 0
    s.holder[transfer] = 2
    s.offset[transfer] = p[transfer] - e2[transfer]
    s.receiver_latched[release1] = False
    s.holder[(s.holder == 2) & (jaw(env, 2) > 0.38)] = 0
    ids = (s.holder > 0).nonzero().flatten()
    if not len(ids):
        return
    tools = torch.where((s.holder[ids] == 1)[:, None], e1[ids], e2[ids])
    desired = tools + s.offset[ids]
    desired[:, 2].clamp_(min=g.REST_Z)
    # Threading constrains the block, not the source tool: excessive stretching releases it.
    stretch = torch.linalg.vector_norm(desired - p[ids], dim=1) > 0.008
    s.holder[ids[stretch]] = 0
    s.receiver_latched[ids[stretch]] = False
    ids, desired = ids[~stretch], desired[~stretch]
    if not len(ids):
        return
    threaded = (s.holder[ids] == 1) & ~s.source_cleared[ids]
    desired[threaded, :2] = s.source[ids[threaded], :2]
    delta = desired - p[ids]
    delta *= (0.004 / torch.linalg.vector_norm(delta, dim=1).clamp_min(0.004))[:, None]
    proposed = p[ids] + delta
    safe = aligned[ids] & g.upright_path_safe(p[ids], proposed)
    s.blocked_moves[ids[~safe]] += 1
    ids, proposed = ids[safe], proposed[safe]
    if not len(ids):
        return
    cleared = (s.holder[ids] == 1) & ~s.source_cleared[ids] & (proposed[:, 2] >= g.CLEAR_Z)
    s.source_cleared[ids[cleared]] = True
    s.offset[ids[cleared], :2] = proposed[cleared, :2] - e1[ids[cleared], :2]
    pose = obj.data.root_pose_w[ids].clone()
    pose[:, :3] = proposed + env.scene.env_origins[ids]
    pose[:, 3:] = pose.new_tensor((1, 0, 0, 0))
    obj.write_root_pose_to_sim(pose, env_ids=ids)
    # Assistance cancels gravity/velocity while held; release is fully dynamic.
    obj.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)


def update(env):
    s = state(env)
    if s.last_step == env.common_step_counter:
        return s
    s.last_step = env.common_step_counter
    p = local_peg(env)
    old = torch.stack([getattr(s, k).clone() for k in METRICS], dim=1)
    s.grasp_psm1 |= s.holder == 1
    speed = torch.linalg.vector_norm(p - s.previous, dim=1) / env.step_dt
    lift = (s.holder == 1) & s.source_cleared & upright(env) & s.previous_valid
    lift &= (p[:, 2] >= g.LIFT_Z) & (p[:, 2] <= 0.080) & (speed < 0.015)
    s.stable_steps = torch.where(lift, s.stable_steps + 1, 0)
    s.lift_stable |= s.grasp_psm1 & (s.stable_steps >= 15)
    s.grasp_psm2 |= s.lift_stable & s.receiver_latched
    received = s.grasp_psm2 & (s.holder == 2) & (jaw(env, 1) > 0.38) & (p[:, 2] >= g.CLEAR_Z)
    s.handover_steps = torch.where(received, s.handover_steps + 1, 0)
    s.handover |= s.handover_steps >= 5  # still held by PSM2 100 ms after donor release
    delta = p - p.new_tensor(g.TARGET)
    obj = env.scene['object'].data
    seated = s.handover & (s.holder == 0) & ~s.receiver_latched & upright(env)
    seated &= hole_on_target(env)
    seated &= delta[:, 2].abs() < 0.0007
    seated &= torch.linalg.vector_norm(obj.root_lin_vel_w, dim=1) < 0.010
    seated &= torch.linalg.vector_norm(obj.root_ang_vel_w, dim=1) < 0.10
    seated &= s.previous_valid & (speed < 0.010) & (jaw(env, 2) > 0.38)
    s.settled_steps = torch.where(seated, s.settled_steps + 1, 0)
    s.placement |= s.settled_steps >= 25  # 0.5 s released and physically settled
    clear = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for a in (1, 2):
        clear &= torch.linalg.vector_norm(ee(env, a) - env.scene.env_origins - p, dim=1) > 0.025
    s.full_success |= (s.settled_steps >= 25) & clear
    now = torch.stack([getattr(s, k) for k in METRICS], dim=1)
    s.events[:] = (now & ~old).float()
    s.previous[:] = p
    s.previous_valid[:] = True
    return s


def success(env):
    s = update(env)
    return getattr(s, {'lift': 'lift_stable', 'handover': 'handover', 'full': 'full_success'}[env.cfg.phase])


def failure(env):
    p = local_peg(env)
    return (p[:, 2] < g.REST_Z - 0.003) | (p[:, 2] > 0.15) | (p[:, 0].abs() > 0.18) | (p[:, 1].abs() > 0.09)


def reward(env):
    """Finite progress budget per stage, one-shot events and a per-step time cost.

    High-watermarks prevent rewarding oscillation, release/regrasp or waiting.
    Terminal phase bonuses are paid once because success immediately resets.
    """
    s = update(env)
    p = local_peg(env)
    e1, e2 = ee(env, 1) - env.scene.env_origins, ee(env, 2) - env.scene.env_origins
    proximity = lambda a, b, scale: torch.exp(-torch.linalg.vector_norm(a - b, dim=1) / scale)
    values = torch.stack([
        proximity(e1, p + p.new_tensor(g.GRASP1), 0.025),
        ((p[:, 2] - g.REST_Z) / (g.LIFT_Z - g.REST_Z)).clamp(0, 1),
        0.4 * proximity(p, p.new_tensor(g.HANDOVER), 0.04)
        + 0.6 * proximity(e2, p + p.new_tensor(g.GRASP2), 0.025),
        ((jaw(env, 1) - 0.18) / 0.20).clamp(0, 1),
        proximity(p, p.new_tensor(g.ABOVE_TARGET), 0.04),
        proximity(p, p.new_tensor(g.TARGET), 0.012),
    ], dim=1)
    gates = torch.stack([~s.grasp_psm1, (s.holder == 1) & ~s.lift_stable,
                         s.lift_stable & (s.holder == 1) & ~s.grasp_psm2,
                         s.grasp_psm2 & ~s.handover, s.handover & (s.holder == 2),
                         s.handover & ~s.full_success], dim=1)
    event_weights = p.new_tensor((2., 4., 4., 6., 10., 40.))
    if env.cfg.phase == 'lift':
        gates[:, 2:] = False
        event_weights[2:] = 0
    elif env.cfg.phase == 'handover':
        gates[:, 4:] = False
        event_weights[4:] = 0
    improvement = (values - s.highwater).clamp_min(0) * gates
    s.highwater[:] = torch.maximum(s.highwater, values * gates)
    total = (improvement * p.new_tensor((2., 4., 4., 2., 4., 4.))).sum(1)
    total += (s.events * event_weights).sum(1)
    total -= 0.02 + 0.0001 * env.action_manager.action.square().sum(1)
    total -= failure(env).float() * 10
    return total / env.step_dt  # RewardManager multiplies by dt


def observation(env):
    """Metre values scaled to centimetres; complete attachment/sequence state visible."""
    s = state(env)
    p = local_peg(env)
    obj = env.scene['object'].data
    features = []
    for a, grasp in ((1, g.GRASP1), (2, g.GRASP2)):
        robot = env.scene[f'robot_{a}'].data
        features.extend([robot.joint_pos - robot.default_joint_pos, robot.joint_vel * 0.1,
                         (ee(env, a) - env.scene.env_origins - p - p.new_tensor(grasp)) * 100])
    features += [p * 100, (p.new_tensor(g.TARGET) - p) * 100,
                 obj.root_quat_w, obj.root_lin_vel_w * 10, obj.root_ang_vel_w * 0.1,
                 s.offset * 100, s.source * 100, (s.previous - p) * 100,
                 torch.stack([getattr(s, k).float() for k in METRICS], dim=1),
                 torch.stack([(s.holder == 1).float(), (s.holder == 2).float(),
                              s.receiver_latched.float(), s.source_cleared.float(),
                              s.previous_valid.float(), s.stable_steps.clamp(max=15).float() / 15,
                              s.handover_steps.clamp(max=5).float() / 5, s.settled_steps.clamp(max=25).float() / 25,
                              s.candidate1.clamp(max=8).float() / 8, s.candidate2.clamp(max=8).float() / 8,
                              env.episode_length_buf.float() / env.max_episode_length], dim=1),
                 s.highwater, env.action_manager.action]
    return torch.cat(features, dim=1)
