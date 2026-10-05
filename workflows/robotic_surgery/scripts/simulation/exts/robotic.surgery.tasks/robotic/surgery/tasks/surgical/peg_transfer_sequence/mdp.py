"""Explicit measured sequence and feedback goals. No simulator imports.

Uses legacy board/post guards and capture thresholds, with a sequence-specific
receiver grasp point. Its real approach still requires Isaac validation.
Guidance is an analytic motor controller, not a learned policy. At guidance=0
all active-arm translation/jaw commands come from SAC; the sequencer still
selects goals, holds the inactive arm and fixes orientation.
"""
import torch
from ..peg_transfer import sac_mdp as physical
from . import geometry as g
from . import assistance

STAGES = ('settle', 'donor_above', 'donor_outside', 'donor_reach', 'donor_close',
          'donor_lift', 'donor_hold', 'donor_transfer', 'receiver_above',
          'receiver_outside', 'receiver_reach', 'receiver_close', 'donor_release',
          'donor_retreat', 'receiver_transport', 'receiver_align', 'receiver_insert',
          'receiver_release', 'receiver_retreat', 'placement_settle')
SKILLS = {'approach': 4, 'grasp': 5, 'lift': 7, 'transfer': 8,
          'receive': 12, 'handover': 14, 'full': 20}
ARMS = (0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 0)


def rotate_inverse(q, v):
    xyz = -q[:, 1:]
    uv = torch.linalg.cross(xyz, v, dim=1)
    return v + 2 * (q[:, :1] * uv + torch.linalg.cross(xyz, uv, dim=1))


class State:
    def __init__(self, env):
        n, d = env.num_envs, env.device
        self.stage = torch.zeros(n, device=d, dtype=torch.long)
        self.age = torch.zeros_like(self.stage)
        self.dwell = torch.zeros_like(self.stage)
        self.targets = torch.zeros(n, 2, 3, device=d)
        self.entered = torch.ones(n, device=d, dtype=torch.bool)
        self.stage_event = torch.zeros(n, device=d)
        self.potential = torch.zeros(n, device=d)
        self.last_blocked = torch.zeros(n, device=d)
        self.last_step = -1


def state(env):
    if not hasattr(env, '_sequence'):
        env._sequence = State(env)
    return env._sequence


def tips(env):
    return torch.stack([physical.ee(env, a) - env.scene.env_origins for a in (1, 2)], dim=1)


def reset(env, env_ids=None):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    s = state(env)
    for name in ('stage', 'age', 'dwell', 'stage_event', 'potential', 'last_blocked'):
        getattr(s, name)[ids] = 0
    s.targets[ids] = tips(env)[ids]
    s.entered[ids] = True


def goals(env):
    s, pstate = state(env), physical.state(env)
    p, ee = physical.local_peg(env), tips(env)
    # Capture measured poses at stage entry, then retain inactive-arm targets.
    s.targets[s.entered] = ee[s.entered]
    s.entered[:] = False
    source = pstate.source.clone()
    source[:, 2] = g.REST_Z
    grasp1 = source + source.new_tensor(g.GRASP1)
    grasp2 = p + p.new_tensor(g.GRASP2)
    def put(index, arm, target):
        ids = s.stage == index
        s.targets[ids, arm - 1] = target[ids]
    put(1, 1, grasp1 + p.new_tensor((0, -.015, .020)))
    put(2, 1, grasp1 + p.new_tensor((0, -.015, 0)))
    put(3, 1, grasp1 + p.new_tensor((0, -.0035, 0)))
    lifted = source.clone(); lifted[:, 2] = .060
    for stage in (5, 6):
        put(stage, 1, lifted - pstate.offset)
    put(7, 1, p.new_tensor(g.HANDOVER).expand_as(p) - pstate.offset)
    put(8, 2, grasp2 + p.new_tensor(g.RECEIVER_OUTSIDE) + p.new_tensor((0, 0, .020)))
    put(9, 2, grasp2 + p.new_tensor(g.RECEIVER_OUTSIDE))
    put(10, 2, grasp2 + p.new_tensor(g.RECEIVER_CAPTURE))
    # Retreat targets must be static: entry pose + displacement, not current pose.
    if not hasattr(s, 'retreat'):
        s.retreat = s.targets.clone()
    for stage, arm, delta in ((13, 1, (0, -.030, .030)), (18, 2, (.030, 0, .035))):
        ids = (s.stage == stage) & (s.age == 0)
        s.retreat[ids, arm - 1] = ee[ids, arm - 1] + p.new_tensor(delta)
        put(stage, arm, s.retreat[:, arm - 1])
    for stage, root in ((14, g.ABOVE_TARGET), (15, (*g.TARGET[:2], g.CLEAR_Z + .002)), (16, g.TARGET)):
        put(stage, 2, p.new_tensor(root).expand_as(p) - pstate.offset)
    return s.targets


def active_arm(env):
    s = state(env)
    return s.stage.new_tensor(ARMS)[s.stage.clamp(max=len(STAGES) - 1)]


def jaw_targets(env):
    stage = state(env).stage
    return torch.stack([torch.where((stage >= 4) & (stage <= 11), -1., 1.),
                        torch.where((stage >= 11) & (stage <= 16), -1., 1.)], dim=1)


def nominal(env):
    """Body-frame metre displacement and jaw target for both arms."""
    error = goals(env) - tips(env)
    cap = torch.where((state(env).stage == 3) | (state(env).stage == 10), .0003, .0012)
    delta = error * .5
    delta *= (cap[:, None] / delta.norm(dim=2).clamp_min(1e-9))[:, :, None].clamp(max=1)
    delta[state(env).stage == 0] = 0
    delta[state(env).stage == 19] = 0
    body = torch.stack([rotate_inverse(env.scene[f'robot_{a}'].data.root_quat_w, delta[:, a-1])
                        for a in (1, 2)], dim=1)
    return body, jaw_targets(env), cap


def expert_action(env):
    """Four-dimensional action that exactly reproduces nominal control at any guidance."""
    body, jaws, cap = nominal(env)
    arm = active_arm(env)
    index = (arm - 1).clamp_min(0)
    ids = torch.arange(env.num_envs, device=env.device)
    alpha = env.cfg.guidance
    scale = (1-alpha)*cap + alpha*env.cfg.residual_translation
    action = torch.cat(((1-alpha)*body[ids, index]/scale[:, None], jaws[ids, index, None]), dim=1)
    action[arm == 0] = 0
    return action.clamp(-1, 1)


def motor_action(env, action):
    """Map 4 continuous commands to real 14-D IK/jaw control; no tool teleportation."""
    body, jaws, cap = nominal(env)
    arm = active_arm(env)
    alpha = env.cfg.guidance
    for a in (1, 2):
        ids = arm == a
        scale = (1-alpha)*cap[ids] + alpha*env.cfg.residual_translation
        body[ids, a-1] = alpha*body[ids, a-1] + scale[:, None]*action[ids, :3]
        body[ids, a-1] *= (cap[ids]/body[ids, a-1].norm(dim=1).clamp_min(1e-9))[:, None].clamp(max=1)
        jaws[ids, a-1] = alpha*jaws[ids, a-1] + (1-alpha)*action[ids, 3]
    out = action.new_zeros(env.num_envs, 14)
    for a, offset in ((1, 0), (2, 7)):
        scale = out.new_tensor(env.action_manager.get_term(f'body_{a}_joint_pos').cfg.scale[:3])
        out[:, offset:offset+3] = body[:, a-1]/scale
        out[:, offset+6] = jaws[:, a-1]
    return out.clamp(-1, 1)


def advance(env):
    s, ps = state(env), physical.update(env)
    if s.last_step == env.common_step_counter:
        return s
    s.last_step = env.common_step_counter
    s.stage_event[:] = 0
    s.age += 1
    p, ee, targets = physical.local_peg(env), tips(env), goals(env)
    errors = (targets-ee).norm(dim=2)
    j1, j2 = physical.jaw(env, 1), physical.jaw(env, 2)
    aligned = env.scene['object'].data.root_quat_w[:, 1:].norm(dim=1) < .005
    near1 = (ee[:, 0]-p-p.new_tensor(g.GRASP1)).norm(dim=1) < g.GRASP_RADIUS
    near2 = (ee[:, 1]-p-p.new_tensor(g.GRASP2)).norm(dim=1) < g.GRASP_RADIUS
    clear = ((ee-p[:,None,:]).norm(dim=2) > .025).all(dim=1)
    fresh_full = (ps.full_success & (ps.settled_steps >= 25) & (ps.holder == 0)
                  & physical.upright(env) & physical.hole_on_target(env) & clear)
    conditions = [s.age >= 25, errors[:, 0] < .0015, errors[:, 0] < .0015,
                  near1 & aligned & (j1 > .38) & (ps.holder == 0), ps.holder == 1,
                  (p[:, 2] >= .058) & ps.source_cleared & (ps.holder == 1), ps.lift_stable & (ps.holder == 1),
                  ((p-p.new_tensor(g.HANDOVER)).norm(dim=1) < .0015) & (ps.holder == 1) & ps.lift_stable,
                  errors[:, 1] < .0015, errors[:, 1] < .0015,
                  near2 & aligned & (j2 > .38) & (ps.holder == 1) & ps.lift_stable,
                  ps.grasp_psm2 & ps.receiver_latched & (ps.holder == 1),
                  ps.handover & (ps.holder == 2) & (j1 > .38),
                  (errors[:, 0] < .002) & (ps.holder == 2) & ps.handover,
                  ((p-p.new_tensor(g.ABOVE_TARGET)).norm(dim=1) < .001) & (ps.holder == 2),
                  (p[:, 2] >= g.CLEAR_Z) & (p[:, 2] < g.CLEAR_Z+.003) & physical.hole_on_target(env) & (ps.holder == 2),
                  ((p-p.new_tensor(g.TARGET)).norm(dim=1) < .0006) & physical.hole_on_target(env) & (ps.holder == 2),
                  (ps.holder == 0) & (j2 > .38) & ps.handover,
                  (errors[:, 1] < .002) & (ps.holder == 0) & ps.handover,
                  fresh_full]
    matrix = torch.stack(conditions, dim=1)
    condition = matrix.gather(1, s.stage.clamp(max=19)[:, None]).squeeze(1) & (s.stage < 20)
    s.dwell = torch.where(condition, s.dwell+1, 0)
    coarse = s.stage.new_tensor([1, 2, 7, 8, 9, 13, 14, 15, 16, 18])
    required = torch.where(torch.isin(s.stage, coarse), 5, 1)
    progressed = s.dwell >= required
    s.stage[progressed] += 1
    s.age[progressed] = 0
    s.dwell[progressed] = 0
    s.entered[progressed] = True
    s.stage_event[progressed] = 1
    goals(env)
    return s


def success(env):
    return advance(env).stage >= SKILLS[env.cfg.skill]


def failure_flags(env):
    s, ps = advance(env), physical.state(env)
    lost = (((s.stage >= 5) & (s.stage <= 11) & (ps.holder != 1)) |
            ((s.stage >= 12) & (s.stage <= 16) & (ps.holder == 0)) |
            ((s.stage >= 13) & (s.stage <= 16) & (ps.holder != 2)))
    ee = tips(env)
    workspace = ((ee[:, :, 0].abs() > .28) | (ee[:, :, 1].abs() > .22) |
                 (ee[:, :, 2] < .003) | (ee[:, :, 2] > .26)).any(dim=1)
    premature = ((s.stage < 4) & (ps.holder != 0)) | ((s.stage < 11) & ps.receiver_latched)
    return dict(peg_out_of_bounds=physical.failure(env), attachment_lost=lost,
                tool_out_of_workspace=workspace, premature_grasp=premature,
                stage_budget_exhausted=s.age >= env.cfg.stage_budget)


def failure(env):
    return torch.stack(list(failure_flags(env).values()), dim=1).any(dim=1)


def observation(env):
    s, ps = state(env), physical.state(env)
    p, ee = physical.local_peg(env), tips(env)
    target = goals(env)
    arms = active_arm(env)
    ids = torch.arange(env.num_envs, device=env.device)
    idx = (arms-1).clamp_min(0)
    body_error = torch.stack([rotate_inverse(env.scene[f'robot_{a}'].data.root_quat_w, target[:, a-1]-ee[:, a-1])
                             for a in (1, 2)], dim=1)
    obj = env.scene['object'].data
    features = [body_error[ids, idx]*100, (target-ee).reshape(env.num_envs, 6)*100,
                p*100, obj.root_quat_w, obj.root_lin_vel_w*10, obj.root_ang_vel_w*.1,
                ee.reshape(env.num_envs, 6)*100]
    for a in (1, 2):
        robot = env.scene[f'robot_{a}'].data
        features.extend([robot.joint_pos-robot.default_joint_pos, robot.joint_vel*.1])
    features += [torch.stack([physical.jaw(env, a) for a in (1, 2)], dim=1),
                 torch.stack([getattr(ps, k).float() for k in physical.METRICS], dim=1),
                 torch.nn.functional.one_hot(ps.holder, 3).float(),
                 torch.stack([ps.receiver_latched, ps.source_cleared], dim=1).float(),
                 ps.offset*100, ps.source*100,
                 torch.stack([ps.stable_steps.clamp(max=15)/15, ps.handover_steps.clamp(max=5)/5,
                              ps.settled_steps.clamp(max=25)/25, ps.candidate1.clamp(max=8)/8,
                              ps.candidate2.clamp(max=8)/8], dim=1),
                 torch.nn.functional.one_hot(s.stage.clamp(max=20), 21).float(),
                 torch.stack([arms == 1, arms == 2], dim=1).float(),
                 p.new_full((env.num_envs, 1), env.cfg.guidance),
                 p.new_full((env.num_envs, 1), SKILLS[env.cfg.skill]/20),
                 (s.age.float()/env.cfg.stage_budget)[:, None], (s.dwell.float()/5)[:, None], s.potential[:, None]]
    return torch.cat(features, dim=1)


def reward(env):
    s, ps = advance(env), physical.state(env)
    ee, target = tips(env), goals(env)
    arms = active_arm(env)
    error = (target-ee).norm(dim=2).gather(1, (arms-1).clamp_min(0)[:, None]).squeeze(1)
    phi = 2*s.stage.float() + torch.exp(-error/.03)*(arms > 0)
    won, failed = success(env), failure(env)
    phi = torch.where(won | failed, 0., phi)
    value = env.cfg.sequence_gamma*phi-s.potential
    s.potential[:] = phi
    weights = value.new_tensor([4., 12., 6., 12., 10., 50.])
    value += (ps.events*weights).sum(dim=1) + s.stage_event*2
    value += won.float()*20 - failed.float()*20
    value -= .005 + .002*env._sequence_action.square().sum(dim=1)
    blocked = (ps.blocked_moves.float()-s.last_blocked).clamp_min(0)
    value -= .2*blocked
    s.last_blocked[:] = ps.blocked_moves
    done = env.reset_buf.nonzero().flatten()
    reasons = failure_flags(env)
    assist_info = assistance.telemetry(env)
    for i in done.tolist():
        env._sequence_pending[i] = dict(stage=int(s.stage[i]), stage_name=STAGES[int(s.stage[i])] if s.stage[i] < 20 else 'complete',
            completed_stages=list(STAGES[:int(s.stage[i])]),
            transfer_zone_reached=bool(s.stage[i] >= 8), reach_psm2=bool(s.stage[i] >= 11),
            donor_released=bool(s.stage[i] >= 13), insertion_reached=bool(s.stage[i] >= 17),
            stage_age=int(s.age[i]), stage_budget_exhausted=bool(s.age[i] >= env.cfg.stage_budget),
            failure_causes=[name for name, mask in reasons.items() if mask[i]],
            assistance={name: values[i] for name, values in assist_info.items()},
            skill=env.cfg.skill, guidance=env.cfg.guidance,
            tool_target_error_mm=(target[i]-ee[i]).norm(dim=1).mul(1000).tolist(),
            peg_root_m=physical.local_peg(env)[i].tolist(),
            holder=int(ps.holder[i]), receiver_latched=bool(ps.receiver_latched[i]),
            source_cleared=bool(ps.source_cleared[i]),
            grasp_distance_mm=[float((ee[i,a-1]-physical.local_peg(env)[i]-ee.new_tensor(grasp)).norm()*1000)
                               for a,grasp in ((1,g.GRASP1),(2,g.GRASP2))],
            peg_quaternion_vector_norm=float(env.scene['object'].data.root_quat_w[i,1:].norm()),
            tool_quaternion_w=[env.scene[f'ee_{a}_frame'].data.target_quat_w[i,0].tolist() for a in (1,2)],
            tool_tip_m=ee[i].tolist(), jaw_rad=[float(physical.jaw(env, a)[i]) for a in (1, 2)])
    return value/env.step_dt
