"""Verified simulator transitions, permanent demo replay and actor imitation.

No simulator imports. SAC critic, entropy and target updates remain SB3 native.
Inference uses the learned actor alone; it never calls the diagnostic controller.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from common import REPO, TASK, asset_hashes, fingerprint, versions

DEMO_SCHEMA = 'peg-sac-demonstrations-1'


def save_demonstrations(path, transitions, episodes, episode_seconds, step_dt):
    path = Path(path)
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(f'Demonstration output already exists: {path}; choose a new name')
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ('observations', 'actions', 'next_observations', 'rewards', 'dones', 'sources', 'stages')
    arrays = {key: np.asarray([row[key] for row in transitions]) for key in fields}
    np.savez_compressed(path, **arrays)
    meta = dict(schema=DEMO_SCHEMA, task=TASK, phase='lift', fingerprint=fingerprint(),
                assets=asset_hashes(), transitions=len(transitions), observation_dim=arrays['observations'].shape[1],
                episode_seconds=episode_seconds, step_dt=step_dt, episodes=episodes,
                controller='capture_probe', assisted_grasp=True, physical_grasp_verified=False,
                versions=versions(), controller_sha256=hashlib.sha256((REPO /
                    'workflows/robotic_surgery/scripts/simulation/scripts/environments/diagnose_peg_grasp.py').read_bytes()).hexdigest(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
    return meta


def load_demonstrations(path, phase='lift'):
    path = Path(path).resolve()
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta.get('schema') != DEMO_SCHEMA or meta.get('task') != TASK or meta.get('phase') != phase:
        raise ValueError('Demonstration schema/task/phase mismatch')
    if meta.get('fingerprint') != fingerprint() or meta.get('assets') != asset_hashes():
        raise ValueError('Demonstrations require the current task code and assets; recollect them')
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta.get('sha256'):
        raise ValueError('Demonstration archive hash mismatch')
    episodes = meta.get('episodes', [])
    if not episodes or any(not e.get('passed') or not e.get('closure_completed_in_capture_zone')
                           or not e.get('metrics', {}).get('grasp_psm1')
                           or not e.get('metrics', {}).get('lift_stable') for e in episodes):
        raise ValueError('Only successful open-arrival/closure/lift episodes can seed training')
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    required = ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'sources', 'stages')
    if any(key not in data for key in required):
        raise ValueError('Incomplete demonstration transitions')
    n = len(data['actions'])
    if not n or any(len(data[key]) != n for key in required):
        raise ValueError('Inconsistent demonstration lengths')
    dim = meta['observation_dim']
    for key, shape in [('observations', (n, dim)), ('next_observations', (n, dim)),
                       ('actions', (n, 14)), ('rewards', (n,)), ('dones', (n,))]:
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f'Invalid demonstration field: {key}')
        data[key] = data[key].astype(np.float32)
    if (np.abs(data['actions']) > 1).any() or not np.isin(data['dones'], [0, 1]).all():
        raise ValueError('Invalid action bounds or terminal flags')
    end = 0
    for episode in episodes:
        start, stop = episode['start'], episode['stop']
        if start != end or stop <= start or stop > n:
            raise ValueError('Invalid demonstration episode boundaries')
        if data['dones'][stop - 1] != 1 or data['dones'][start:stop - 1].any():
            raise ValueError('Each successful demonstration needs exactly one real terminal transition')
        if not (data['sources'][start:stop] == episode['source']).all():
            raise ValueError('Demonstration source labels disagree with episode metadata')
        if not np.array_equal(data['next_observations'][start:stop - 1], data['observations'][start + 1:stop]):
            raise ValueError('Demonstration observations are not a continuous trajectory')
        end = stop
    if end != n or meta.get('transitions') != n:
        raise ValueError('Demonstration episode coverage mismatch')
    data['groups'] = [np.flatnonzero((data['sources'] == source) & (data['stages'] == stage))
                      for source in np.unique(data['sources']) for stage in np.unique(data['stages'])]
    data['groups'] = [group for group in data['groups'] if len(group)]
    return data, meta


def sample_ids(data, count):
    # Balance source and stage: long approaches must not drown out jaw closure.
    groups = data['groups']
    chosen = np.random.randint(len(groups), size=count)
    return np.array([np.random.choice(groups[group]) for group in chosen])


class DemonstrationReplayBuffer(ReplayBuffer):
    def __init__(self, *args, demo_path, demo_fraction=0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_path, self.demo_fraction = str(demo_path), demo_fraction
        self._demos = None

    def sample(self, batch_size, env=None):
        if self._demos is None:
            self._demos, _ = load_demonstrations(self.demo_path)
        count = batch_size if self.size() == 0 else max(1, round(batch_size * self.demo_fraction))
        ids = sample_ids(self._demos, count)
        demo = ReplayBufferSamples(
            self.to_torch(self._normalize_obs(self._demos['observations'][ids], env)),
            self.to_torch(self._demos['actions'][ids]),
            self.to_torch(self._normalize_obs(self._demos['next_observations'][ids], env)),
            self.to_torch(self._demos['dones'][ids, None]),
            self.to_torch(self._normalize_reward(self._demos['rewards'][ids, None], env)),
        )
        if count == batch_size:
            return demo
        online = super().sample(batch_size - count, env)
        # SB3 2.9 adds an optional discounts field; one-step replay returns None.
        return ReplayBufferSamples(*(None if a is None and b is None else torch.cat((a, b))
                                     for a, b in zip(online, demo)))


def imitate_actor(model, data, updates, batch_size=256, weight=1.0):
    losses = []
    model.policy.set_training_mode(True)
    # A 0.01 normalized translation error is 60 micrometres. Jaw and rotation
    # losses alone would overwhelm the small translation commands in the probe.
    weights = torch.tensor([100, 100, 100, 25, 25, 25, 1] * 2, device=model.device)
    # Keep initial exploration near the demonstrated path, also in unused axes.
    target_std = torch.tensor([-5.3] * 6 + [-2.3] + [-5.3] * 6 + [-2.3], device=model.device)
    for _ in range(updates):
        ids = sample_ids(data, batch_size)
        obs = torch.as_tensor(data['observations'][ids], device=model.device)
        target = torch.as_tensor(data['actions'][ids], device=model.device).clamp(-0.995, 0.995)
        mean, log_std, _ = model.actor.get_action_dist_params(obs)
        loss = weight * (((mean.tanh() - target).square() * weights).mean()
                         + 0.01 * (log_std - target_std).square().mean())
        model.actor.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 10.)
        model.actor.optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses)) if losses else 0.0


class DemonstrationSAC(SAC):
    """Native SAC updates followed by an auxiliary supervised actor update."""
    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['_demo_data']

    def train(self, gradient_steps, batch_size=64):
        super().train(gradient_steps, batch_size)
        if not hasattr(self, '_demo_data'):
            self._demo_data, _ = load_demonstrations(self.replay_buffer.demo_path)
        loss = imitate_actor(self, self._demo_data, gradient_steps, batch_size, self.bc_weight)
        self.logger.record('train/demonstration_actor_loss', loss)
