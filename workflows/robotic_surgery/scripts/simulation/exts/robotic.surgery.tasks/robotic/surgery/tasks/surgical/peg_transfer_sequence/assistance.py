"""Guarded pose servo for the explicitly assisted sequence task, not a force grasp.

Keep a last safe upright reference while held. Minor PhysX contact disturbances
must not disable the servo and leave gravity active. Initial capture stays at the
legacy strict threshold; board/post sweep guards and 8 mm stretch release remain.
Large disturbances release rather than being silently projected away.
"""
import torch
from ..peg_transfer import sac_mdp as physical
from . import geometry as g

BREAK_REASONS = ('none', 'tool_stretch', 'large_rotation', 'large_displacement', 'unsafe_restoration')


def state(env):
    s = physical.state(env)
    if not hasattr(s, 'held_reference'):
        n, d = env.num_envs, env.device
        s.held_reference = torch.zeros(n, 3, device=d)
        s.reference_valid = torch.zeros(n, dtype=torch.bool, device=d)
        s.contact_corrections = torch.zeros(n, dtype=torch.long, device=d)
        s.break_reason = torch.zeros(n, dtype=torch.long, device=d)
        s.raw_rotation_peak = torch.zeros(n, device=d)
        s.raw_displacement_peak = torch.zeros(n, device=d)
    return s


def begin_step(env):
    s = state(env)
    s.raw_rotation_peak.zero_()
    s.raw_displacement_peak.zero_()


def assist(env):
    s, obj = state(env), env.scene['object']
    p = physical.local_peg(env).clone()
    e1, e2 = (physical.ee(env, a) - env.scene.env_origins for a in (1, 2))
    closed1, closed2 = physical.jaw(env, 1) < .18, physical.jaw(env, 2) < .18
    near1 = (e1-p-p.new_tensor(g.GRASP1)).norm(dim=1) < g.GRASP_RADIUS
    near2 = (e2-p-p.new_tensor(g.GRASP2)).norm(dim=1) < g.GRASP_RADIUS
    raw_rotation = obj.data.root_quat_w[:, 1:].norm(dim=1)
    aligned = raw_rotation < .005  # Unchanged initial capture threshold.
    was_held = (s.holder > 0) & s.reference_valid
    reference = torch.where(was_held[:, None], s.held_reference, p)
    # A transient collision displacement must not fake receiver capture outside
    # the held reference's original 4 mm grasp zone.
    near2 &= (e2-reference-p.new_tensor(g.GRASP2)).norm(dim=1) < g.GRASP_RADIUS
    displacement = (p-reference).norm(dim=1)
    s.raw_rotation_peak[:] = torch.maximum(s.raw_rotation_peak, torch.where(was_held, raw_rotation, 0.))
    s.raw_displacement_peak[:] = torch.maximum(s.raw_displacement_peak, displacement)

    candidate1 = (s.holder == 0) & ~s.handover & closed1 & near1 & aligned
    s.candidate1 = torch.where(candidate1, s.candidate1+1, 0)
    attach = candidate1 & (s.candidate1 >= 8)
    s.holder[attach] = 1
    s.offset[attach] = p[attach]-e1[attach]
    s.break_reason[attach] = 0
    candidate2 = (s.holder == 1) & s.lift_stable & s.source_cleared & closed2 & near2
    s.candidate2 = torch.where(candidate2, s.candidate2+1, 0)
    s.receiver_latched |= candidate2 & (s.candidate2 >= 8)
    s.receiver_latched &= closed2 & near2 & (s.holder == 1)
    release1 = (s.holder == 1) & (physical.jaw(env, 1) > .38)
    transfer = release1 & s.receiver_latched & s.grasp_psm2
    s.holder[release1] = 0
    s.holder[transfer] = 2
    s.offset[transfer] = reference[transfer]-e2[transfer]  # No jump between holders.
    s.receiver_latched[release1] = False
    s.holder[(s.holder == 2) & (physical.jaw(env, 2) > .38)] = 0
    s.reference_valid[s.holder == 0] = False
    ids = (s.holder > 0).nonzero().flatten()
    if not len(ids):
        return
    tools = torch.where((s.holder[ids] == 1)[:, None], e1[ids], e2[ids])
    desired = tools+s.offset[ids]
    desired[:, 2].clamp_(min=g.REST_Z)
    reason = torch.zeros(len(ids), dtype=torch.long, device=env.device)
    reason[(desired-reference[ids]).norm(dim=1) > .008] = 1
    reason[raw_rotation[ids] > .04] = 2  # No correction of grossly tilted poses.
    reason[displacement[ids] > .004] = 3  # Restoration also obeys the 4 mm sweep resolution.
    restoration_safe = g.upright_path_safe(p[ids], reference[ids])
    reason[(reason == 0) & ~restoration_safe] = 4
    broken = reason > 0
    s.break_reason[ids[broken]] = reason[broken]
    s.holder[ids[broken]] = 0
    s.receiver_latched[ids[broken]] = False
    s.reference_valid[ids[broken]] = False
    ids, desired = ids[~broken], desired[~broken]
    if not len(ids):
        return
    threaded = (s.holder[ids] == 1) & ~s.source_cleared[ids]
    desired[threaded, :2] = s.source[ids[threaded], :2]
    delta = desired-reference[ids]
    delta *= (.004/delta.norm(dim=1).clamp_min(.004))[:, None]
    proposed = reference[ids]+delta
    safe = g.upright_path_safe(reference[ids], proposed)
    s.blocked_moves[ids[~safe]] += 1
    # A rejected command retains the last safe upright pose, never free falls.
    proposed = torch.where(safe[:, None], proposed, reference[ids])
    cleared = (s.holder[ids] == 1) & ~s.source_cleared[ids] & (proposed[:, 2] >= g.CLEAR_Z)
    s.source_cleared[ids[cleared]] = True
    s.offset[ids[cleared], :2] = proposed[cleared, :2]-e1[ids[cleared], :2]
    s.contact_corrections[ids] += ((raw_rotation[ids] >= .005) | (displacement[ids] > .0005)).long()
    pose = obj.data.root_pose_w[ids].clone()
    pose[:, :3] = proposed+env.scene.env_origins[ids]
    pose[:, 3:] = pose.new_tensor((1., 0., 0., 0.))
    obj.write_root_pose_to_sim(pose, env_ids=ids)
    obj.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), env_ids=ids)
    s.held_reference[ids] = proposed
    s.reference_valid[ids] = True


def telemetry(env):
    s = state(env)
    return dict(holder=s.holder.tolist(), receiver_latched=s.receiver_latched.tolist(),
        blocked_moves=s.blocked_moves.tolist(), contact_corrections=s.contact_corrections.tolist(),
        break_reason=[BREAK_REASONS[int(k)] for k in s.break_reason],
        raw_rotation_peak=s.raw_rotation_peak.tolist(),
        raw_displacement_peak_mm=(s.raw_displacement_peak*1000).tolist())
