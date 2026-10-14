"""Residual motors around a frozen checkpoint13; no teacher at inference."""
from dataclasses import dataclass, asdict
import numpy as np
import torch
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env import VecEnvWrapper

CACHE = ('selected_option', 'selected_goal', 'selected_prediction', 'current_goal', 'decision_scores')
SCHEMA = 'peg-bc-residual-sac-1'
OBS_DIM = 241
ACTION_DIM = 14


@dataclass(frozen=True)
class ResidualLimits:
    translation: float = .05  # norm bound: 0.15 mm of the 3 mm motor request
    rotation: float = 0.
    jaw: float = 0.
    energy_cost: float = .002

    def __post_init__(self):
        if not 0 <= self.translation <= .1 or self.rotation != 0 or self.jaw != 0:
            raise ValueError('First curriculum permits bounded free translation only')
        if not np.isfinite(self.energy_cost) or self.energy_cost < 0:
            raise ValueError('Residual energy cost must be finite and nonnegative')


def free_mask(state, profile):
    # Geometric clearance, independent of teacher stages and BC option numbers.
    from rl_core import apply, conjugate
    local = apply(conjugate(state['peg_q'])[:, None].expand(-1, 2, -1),
                  state['tip'] - state['peg'][:, None])
    bounds = local.new_tensor(profile['bounds_m'])
    nearest = torch.maximum(bounds[0], torch.minimum(local, bounds[1]))
    clearance = (local - nearest).norm(dim=-1)
    peak = state['forces'].norm(dim=-1).amax(-1)
    return ((clearance - .006) / .019).clamp(0, 1) * (peak < .002)


def mix_actions(base, residual, mask, limits):
    if base.shape != residual.shape or base.shape[-1] != ACTION_DIM:
        raise ValueError('Expected matching 14-dimensional actions')
    if not torch.isfinite(residual).all():
        raise ValueError('Nonfinite residual action')
    shaped = residual.clamp(-1, 1).reshape(-1, 2, 7)
    position = shaped[..., :3]
    position = position / position.norm(dim=-1, keepdim=True).clamp_min(1)
    delta = torch.zeros_like(shaped)
    delta[..., :3] = position * limits.translation * mask[..., None]
    executed = (base + delta.flatten(1)).clamp(-1, 1)
    return executed, executed - base


class ResidualEnv(VecEnvWrapper):
    """Replay stores residual actions and the PRE-reset augmented terminal state."""
    def __init__(self, vec, base_policy, guard, limits=ResidualLimits()):
        super().__init__(vec, observation_space=Box(-np.inf, np.inf, (OBS_DIM,), np.float32))
        self.base = base_policy.eval().requires_grad_(False)
        self.guard = guard
        self.limits = limits
        self.device = vec.device
        self.terminal_augmented = None
        self.correction_peak = 0.
        self.active = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        update = vec.progress.update

        def capture_update(state, action, skill='full'):
            result = update(state, action, skill)
            if bool((result[1] | result[2] | (vec.age >= vec.episode_steps)).any()):
                from rl_core import observe
                saved = {k: getattr(self.base, k, None) for k in CACHE}
                # forward assigns new cache tensors; clones also protect future in-place changes.
                for k, value in saved.items():
                    if isinstance(value, torch.Tensor): setattr(self.base, k, value.clone())
                try:
                    self.terminal_augmented = self.prepare(observe(state, vec.reference, vec.progress))
                finally:
                    for k, value in saved.items(): setattr(self.base, k, value)
            return result

        vec.progress.update = capture_update

    @torch.no_grad()
    def prepare(self, raw):
        from autonomy_learning import physical_observation
        raw = torch.as_tensor(raw, device=self.device, dtype=torch.float32)
        self.next_base = self.base(physical_observation(raw)).detach()
        choice = torch.nn.functional.one_hot(self.base.selected_option, 19).float()
        vec = self.venv
        ref, progress, support = vec.reference, vec.progress, vec.support
        parts = [raw, self.next_base, choice]
        parts += [getattr(self.base, k) for k in CACHE[1:4]]
        parts += [ref.grip_age, ref.speed_cap / .1, ref.anchor.flatten(1) / .03, ref.anchor_q.flatten(1),
                  progress.anchor.flatten(1) / .03, progress.anchor_q.flatten(1),
                  progress.contact_age / .4, progress.lift_age[:, None] / 2,
                  progress.place_age[:, None], progress.receiver_alone_age[:, None] / 2,
                  torch.nn.functional.one_hot(support.holder.long(), 3).float(),
                  support.relative / .03, support.relative_q, support.acquire_jaw[:, None] / .5,
                  support.placement_active[:, None].float(), vec.age[:, None].float() / vec.episode_steps]
        obs = torch.cat(parts, 1)
        if obs.shape[1] != OBS_DIM or not torch.isfinite(obs).all():
            raise ValueError(f'Invalid augmented observation: {obs.shape}')
        # Unbounded histories are represented but clipped for a stable MLP input.
        return obs.clamp(-20, 20).cpu().numpy()

    def reset(self):
        self.base.reset()
        self.active.fill_(True)
        return self.prepare(self.venv.reset())

    def step_async(self, actions):
        self.pending = actions

    @torch.no_grad()
    def step_wait(self):
        vec = self.venv
        residual = torch.as_tensor(self.pending, device=self.device, dtype=torch.float32)
        executed, delta = mix_actions(self.next_base, residual, free_mask(vec.state, vec.profile), self.limits)
        self.correction_peak = max(self.correction_peak, float(delta.abs().max()))
        executed = self.guard.apply(executed, vec.state, vec.reference)
        executed = torch.where(self.active[:, None], executed, torch.zeros_like(executed))
        conditions = [dict(row) for row in vec.reset_records]
        self.terminal_augmented = None
        # prepare() during the pre-reset terminal hook previews the next base action;
        # prepare() below overwrites this cache with the real next mixed-batch action.
        raw, reward, dones, infos = vec.step(executed.cpu().numpy())
        reward = reward - self.limits.energy_cost * delta.square().mean(-1).cpu().numpy()
        ids = np.flatnonzero(dones)
        for i in ids:
            if self.terminal_augmented is None: raise RuntimeError('Missing terminal context')
            infos[i]['terminal_observation'] = self.terminal_augmented[i].copy()
            infos[i]['metrics']['reset'] = conditions[i]
        if len(ids): self.base.reset(torch.as_tensor(ids, device=self.device))
        return self.prepare(raw), reward, dones, infos

    def evidence(self):
        return dict(schema=SCHEMA, observation_dim=OBS_DIM, action_dim=ACTION_DIM,
                    residual_limits=asdict(self.limits), correction_peak=self.correction_peak,
                    frozen_bc=True, teacher_at_inference=False, assisted_grasp=True,
                    native_full_policy_verified=False, contact_rotation_jaw_corrections=False,
                    mask='unloaded arm; peg mesh AABB clearance ramp 6..25 mm')
