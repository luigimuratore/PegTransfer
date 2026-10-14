"""Measured, goal-conditioned BC for both arms across the complete sequence.

The network learns positive Cartesian and jaw-feedback gains. Stage selection,
orientation/IK, contact guards and the explicitly reported fixture remain code.
No clock, reward, teacher action or future state enters the actor observation.
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from full_task import STAGES
from learning import PHYSICS_FILES
from motion import rotation

SCHEMA = 'peg-full-motor-bc-1'
OBS_DIM = 76
ACTION_DIM = 5
PHYSICS_NAMES = tuple(dict.fromkeys((*PHYSICS_FILES, 'kinematics.py', 'full_geometry.py',
    'handover_grip.py', 'holding_compliance.py', 'peg_support.py')))
CONTACT_STAGES = ('donor_reach', 'donor_close', 'donor_lift', 'donor_hold', 'donor_transfer',
    'receiver_reach', 'receiver_close', 'receiver_hold', 'receiver_transport', 'receiver_insert')
SLOW_STAGES = ('donor_lift', 'donor_hold', 'donor_transfer', 'receiver_hold', 'receiver_transport')


def code_hash(path):
    """Ignore documentation changes, preserve every executable statement."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                body.pop(0)
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def physics_hashes(folder):
    return {name: (code_hash(folder / name) if name.endswith('.py') else
        hashlib.sha256((folder / name).read_bytes()).hexdigest()) for name in PHYSICS_NAMES}


def speeds(parameters):
    return np.array([parameters['lift_speed'] if name in SLOW_STAGES else
        parameters['contact_speed'] if name in CONTACT_STAGES else parameters['free_speed']
        for name in STAGES], dtype=np.float32)


def active_arm(stage):
    return 0 if STAGES[int(stage)].startswith('donor') else 1


def observe(rows, stage, goals, goal_quaternions, references, jaw_commands, release_starts):
    values = []
    for arm, row in enumerate(rows):
        error_rot = (rotation(goal_quaternions[arm]) * rotation(row['tool_quaternion_wxyz']).inv()).as_rotvec()
        values.extend(np.r_[(np.asarray(goals[arm]) - row['tool_tip_w_m']) / .05, error_rot,
            (np.asarray(references[arm]) - row['tool_tip_w_m']) / .003,
            np.asarray(row['peg_in_tool_m']) / .02, row['peg_quat_in_tool'],
            np.asarray(row['finger_force_vectors_n']).reshape(-1), np.asarray(row['jaw_rad']) / .5,
            jaw_commands[arm] / .5, row['peg_bottom_m'] / .05, row['peg_speed_m_s'] / .02])
    result = np.r_[values, np.eye(len(STAGES))[STAGES.index(stage)], np.asarray(release_starts) / .5]
    if result.shape != (OBS_DIM,) or not np.isfinite(result).all():
        raise ValueError('Invalid full-policy observation')
    return result.astype(np.float32)


def decode_tensor(obs, latent, parameters):
    """Return executed reference/jaw increments, normalized by speed and dt."""
    import torch
    stage = obs[..., 54:74].argmax(dim=-1)
    donor = torch.as_tensor([name.startswith('donor') for name in STAGES], device=obs.device)[stage]
    head = torch.where(donor[..., None], obs[..., :27], obs[..., 27:54])
    stage_speeds = torch.as_tensor(speeds(parameters), device=obs.device)
    speed = stage_speeds[stage]
    contact = torch.as_tensor([name in CONTACT_STAGES for name in STAGES], device=obs.device)[stage]
    lead = head[..., 6:9] * .003
    error = head[..., :3] * .05 - lead
    velocity = (5. + 47.5 * (1. + latent[..., :3].clamp(-1., 1.))) * error
    velocity = velocity * torch.clamp(speed / velocity.norm(dim=-1).clamp_min(1e-12), max=1.)[..., None]
    proposed = lead + .01 * velocity
    limit = torch.where(contact, .0005, .003)
    projected = proposed * torch.clamp(limit / proposed.norm(dim=-1).clamp_min(1e-12), max=1.)[..., None]
    translation = (projected - lead) / (.01 * speed[..., None])
    jaws = []
    for arm, offset in ((0, 0), (1, 27)):
        force = obs[..., offset+16:offset+22].reshape(*obs.shape[:-1], 2, 3).norm(dim=-1)
        low, high = force.amin(dim=-1), force.amax(dim=-1)
        previous = obs[..., offset+24] * .5
        closing = stage == (4 if arm == 0 else 11)
        holding = ((stage >= 5) & (stage <= 11)) if arm == 0 else ((stage >= 12) & (stage <= 16))
        gripping = closing | holding
        rate = torch.where(high > .03, .02/.15, 1.)
        if arm == 1:
            rate = torch.where(closing, .02/.15, rate)
        base = torch.where(low < parameters['grip_force']-.05, -rate,
            torch.where(low > parameters['grip_force']+.05, .05/.15, 0.))
        if arm == 1:
            imbalance = (force[..., 1]-force[..., 0]) / force.sum(dim=-1).clamp_min(1e-9)
            balancing = closing & (high > .002) & (imbalance.abs() > .2)
            base = torch.where(balancing, 0., base)
        base = torch.where(gripping, base, 0.)
        release = stage == (12 if arm == 0 else 17)
        release_start = obs[..., 74+arm] * .5
        opening = release & ((high >= .01) | (previous < release_start+.02))
        base = torch.where(opening, 1., base)
        gain = 1. + .5 * latent[..., 3+arm].clamp(-1., 1.)
        velocity_jaw = (base * gain).clamp(-1., 1.)
        upper = torch.where(gripping, parameters['donor_open'] if arm == 0 else parameters['receiver_open'], .5)
        target = torch.maximum(torch.zeros_like(previous), torch.minimum(previous+.0015*velocity_jaw, upper))
        jaws.append((target-previous) / .0015)
    return torch.cat([translation, *[jaw[..., None] for jaw in jaws]], dim=-1)


def export(report_path, output, base_manifest):
    from audit_full import audit
    from replay_full import verify
    checked = audit(report_path)
    report = json.loads(report_path.read_text())
    preserved, bundle = verify(base_manifest)
    if checked['full_success'] < 3 or not checked['assisted_grasp']:
        raise ValueError('First full dataset requires at least three verified assisted successes')
    if report['source_hashes'] != json.loads((bundle/'reference.json').read_text())['source_hashes']:
        raise ValueError('Collection differs from the preserved successful teacher')
    params = preserved['parameters']
    records = []; previous = None; refs = quats = jaws = releases = None
    with Path(report['trace']).open() as trace:
        for line in trace:
            row = json.loads(line); rows = row['pre_action_state']; stage = row['stage']
            arm = active_arm(STAGES.index(stage))
            new_episode = previous is None or previous['episode'] != row['episode']
            new_stage = new_episode or previous['stage'] != stage
            if new_episode:
                refs = [np.array(r['tool_tip_w_m']) for r in rows]
                quats = [np.array(r['tool_quaternion_wxyz']) for r in rows]
                jaws = np.array([params['donor_open'], params['receiver_open']])
            if new_stage:
                if not new_episode:
                    last = active_arm(STAGES.index(previous['stage']))
                    refs[last] = np.array(previous[f'psm{last+1}']['tool_tip_w_m'])
                    quats[last] = np.array(previous[f'psm{last+1}']['tool_quaternion_wxyz'])
                refs[arm] = np.array(rows[arm]['tool_tip_w_m'])
                releases = jaws.copy()
            goals = [v.copy() for v in refs]; goalq = [v.copy() for v in quats]
            goals[arm] = np.array(row['phase_target_position_w_m'])
            goalq[arm] = np.array(row['phase_target_quaternion_wxyz'])
            obs = observe(rows, stage, goals, goalq, refs, jaws, releases)
            next_refs = np.asarray(row['commanded_positions_w_m'])
            next_jaws = np.asarray(row['jaw_targets_rad'])
            speed = speeds(params)[STAGES.index(stage)]
            label = np.r_[(next_refs[arm]-refs[arm])/(.01*speed), (next_jaws-jaws)/.0015]
            records.append((obs, label.astype(np.float32), row['episode'], STAGES.index(stage), bool(row['motion_allowed'])))
            refs = [v.copy() for v in next_refs]
            quats = [np.array(v) for v in row['commanded_quaternions_wxyz']]
            jaws = next_jaws.copy(); previous = row
    metadata = dict(schema=SCHEMA, report=str(report_path.resolve()), report_sha256=checked['report_sha256'],
        trace_sha256=checked['trace_sha256'], base_manifest=str(base_manifest.resolve()),
        base_manifest_sha256=hashlib.sha256(base_manifest.read_bytes()).hexdigest(),
        parameters=params, physics_hashes=physics_hashes(Path(report['source_snapshot'])),
        episodes=list(range(report['episodes'])), transitions=len(records), stages=list(STAGES),
        assisted_grasp=True, full_sequences_verified=True, source='L5', target='R2', control_dt_s=.01,
        input_scope='measured current state, goals, previous actuator references and stage; no clock, reward or future state')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as file:
        np.savez_compressed(file, observations=np.stack([r[0] for r in records]),
            actions=np.stack([r[1] for r in records]), episode=np.array([r[2] for r in records]),
            stage=np.array([r[3] for r in records]), motion_allowed=np.array([r[4] for r in records]),
            metadata=json.dumps(metadata))
    return metadata


def load_data(path):
    with np.load(path, allow_pickle=False) as file:
        data = {name:file[name].copy() for name in ('observations','actions','episode','stage','motion_allowed')}
        meta = json.loads(str(file['metadata']))
    count = meta['transitions']
    if meta['schema'] != SCHEMA or not meta['full_sequences_verified'] or meta['stages'] != list(STAGES):
        raise ValueError('Dataset is not a verified compatible full collection')
    for name, shape in (('observations',(count,OBS_DIM)),('actions',(count,ACTION_DIM)),
        ('episode',(count,)),('stage',(count,)),('motion_allowed',(count,))):
        if data[name].shape != shape or not np.isfinite(data[name]).all():
            raise ValueError(f'Invalid full dataset array: {name}')
    if not np.issubdtype(data['stage'].dtype,np.integer) or not np.issubdtype(data['episode'].dtype,np.integer):
        raise ValueError('Integer stage and episode IDs required')
    if ((data['stage'] < 0) | (data['stage'] >= len(STAGES))).any():
        raise ValueError('Invalid stage IDs')
    if not np.array_equal(data['observations'][:,54:74], np.eye(len(STAGES))[data['stage']]):
        raise ValueError('Actor stage features differ from labels')
    if set(np.unique(data['episode'])) != set(meta['episodes']) or len(meta['episodes']) < 3:
        raise ValueError('At least three complete independent episodes required')
    for episode in meta['episodes']:
        if set(data['stage'][data['episode'] == episode]) != set(range(len(STAGES))):
            raise ValueError('Incomplete full-task episode')
    return data, meta


class FullMotorPolicy:
    def __init__(self, path, device='cuda:0'):
        import torch
        from torch_backend import resolve_device
        self.path = Path(path)
        self.device = resolve_device(device)
        with np.load(path, allow_pickle=False) as file:
            self.metadata = json.loads(str(file['metadata']))
            if self.metadata['schema'] != SCHEMA:
                raise ValueError('Wrong full-policy schema')
            self.mean = torch.as_tensor(file['mean'], device=self.device)
            self.std = torch.as_tensor(file['std'], device=self.device)
            self.layers = [(torch.as_tensor(file[f'w{i}'],device=self.device),
                torch.as_tensor(file[f'b{i}'],device=self.device)) for i in range(3)]
        if self.mean.shape != (OBS_DIM,) or self.std.shape != (OBS_DIM,) or not (self.std > 0).all():
            raise ValueError('Invalid full-policy normalization')
        previous = OBS_DIM
        for index,(weight,bias) in enumerate(self.layers):
            width = ACTION_DIM if index == 2 else 256
            if weight.shape != (width,previous) or bias.shape != (width,):
                raise ValueError('Invalid full-policy layer dimensions')
            previous = width
        if not all(torch.isfinite(a).all() for a in (self.mean,self.std,*[a for pair in self.layers for a in pair])):
            raise ValueError('Non-finite full-policy weights')

    def validate(self, args):
        if physics_hashes(Path(__file__).parent) != self.metadata['physics_hashes']:
            raise ValueError('Physical sources differ from the full-policy dataset')
        for key,value in self.metadata['parameters'].items():
            if key not in ('seed','stage_steps','camera_eye','camera_target') and getattr(args,key) != value:
                raise ValueError(f'Full-policy physical/control parameter differs: {key}')

    def predict(self, obs):
        import torch
        obs = np.asarray(obs,dtype=np.float32)
        if obs.shape != (OBS_DIM,) or not np.isfinite(obs).all():
            raise ValueError('Invalid full-policy input')
        with torch.inference_mode():
            raw = torch.as_tensor(obs,device=self.device)
            x = (raw-self.mean)/self.std
            for index,(weight,bias) in enumerate(self.layers):
                x = torch.nn.functional.linear(x,weight,bias)
                x = x.tanh() if index == 2 else x.relu()
            return decode_tensor(raw,x,self.metadata['parameters']).cpu().numpy()
