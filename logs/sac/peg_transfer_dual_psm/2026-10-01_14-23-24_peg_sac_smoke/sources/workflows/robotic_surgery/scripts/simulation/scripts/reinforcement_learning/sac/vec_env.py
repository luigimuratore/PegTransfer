"""SB3 VecEnv for the SAC task, preserving pre-reset observations and evidence.

Independent of Isaac imports so the replay boundary can be tested on CPU.
"""

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv


class PegSACVecEnv(VecEnv):
    def __init__(self, env):
        self.env = env
        base = env.unwrapped
        self.sim_device = base.device
        space = base.single_observation_space['policy']
        if not isinstance(space, gym.spaces.Box):
            raise TypeError('SAC requires a concatenated policy observation')
        dim = base.single_action_space.shape
        if dim != (14,):
            raise ValueError(f'Expected 14 continuous actions, found {dim}')
        super().__init__(base.num_envs, space, gym.spaces.Box(-1., 1., dim, dtype=np.float32))
        self.returns = np.zeros(self.num_envs)
        self.lengths = np.zeros(self.num_envs, dtype=int)

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self):
        obs, _ = self.env.reset()
        self.returns[:] = 0
        self.lengths[:] = 0
        return obs['policy'].detach().cpu().numpy().copy()

    def step_async(self, actions):
        self.actions = torch.as_tensor(actions, device=self.sim_device, dtype=torch.float32).clamp(-1, 1)

    def step_wait(self):
        obs, reward, terminated, truncated, _ = self.env.step(self.actions)
        obs = obs['policy'].detach().cpu().numpy().copy()
        reward = reward.detach().cpu().numpy().copy()
        terminated = terminated.cpu().numpy()
        truncated = truncated.cpu().numpy()
        done = terminated | truncated
        self.returns += reward
        self.lengths += 1
        infos = [{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(done):
            saved = self.unwrapped._sac_terminal[i]
            infos[i] = {
                'terminal_observation': saved['observation'].copy(),
                'TimeLimit.truncated': bool(truncated[i] and not terminated[i]),
                'episode': {'r': float(self.returns[i]), 'l': int(self.lengths[i])},
                'transfer': {k: v for k, v in saved.items() if k != 'observation'},
                'is_success': saved['metrics']['full_success'],
            }
        self.returns[done] = 0
        self.lengths[done] = 0
        return obs, reward, done, infos

    def close(self):
        self.env.close()

    def get_attr(self, attr_name, indices=None):
        ids = self._get_indices(indices)
        value = getattr(self.unwrapped, attr_name)
        if isinstance(value, torch.Tensor) and value.ndim:
            return [value[i].detach().cpu().numpy() for i in ids]
        return [value for _ in ids]

    def set_attr(self, attr_name, value, indices=None):
        if indices is not None:
            raise NotImplementedError('Per-environment attribute mutation is unsupported')
        setattr(self.unwrapped, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self.env, method_name)(*args, **kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def get_images(self):
        return [self.env.render()]
