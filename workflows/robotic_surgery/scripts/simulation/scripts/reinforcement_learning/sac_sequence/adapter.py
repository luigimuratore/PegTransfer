"""Four-command SB3 vector adapter with true pre-reset terminal observations."""
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import VecEnv

class SequenceVecEnv(VecEnv):
    def __init__(self, env):
        self.env = env
        base = env.unwrapped
        self.sim_device = base.device
        if base.single_action_space.shape != (4,):
            raise ValueError('Sequence task must expose four continuous active-arm actions')
        super().__init__(base.num_envs, base.single_observation_space['policy'],
                         gym.spaces.Box(-1., 1., (4,), dtype=np.float32))
        self.returns = np.zeros(self.num_envs)
        self.lengths = np.zeros(self.num_envs, dtype=int)
    @property
    def unwrapped(self):
        return self.env.unwrapped
    def reset(self):
        obs, _ = self.env.reset()
        self.returns[:] = 0; self.lengths[:] = 0
        return obs['policy'].detach().cpu().numpy().copy()
    def step_async(self, actions):
        self.actions = torch.as_tensor(actions, device=self.sim_device, dtype=torch.float32).clamp(-1, 1)
    def step_wait(self):
        obs, reward, term, trunc, _ = self.env.step(self.actions)
        reward = reward.detach().cpu().numpy().copy()
        term, trunc = term.cpu().numpy(), trunc.cpu().numpy()
        done = term | trunc
        self.returns += reward; self.lengths += 1
        infos = [{} for _ in range(self.num_envs)]
        for i in np.flatnonzero(done):
            saved = self.unwrapped._sac_terminal[i]
            infos[i] = dict(terminal_observation=saved['observation'].copy(),
                transfer={k: v for k, v in saved.items() if k != 'observation'},
                episode=dict(r=float(self.returns[i]), l=int(self.lengths[i])),
                is_success=saved['phase_success'])
            infos[i]['TimeLimit.truncated'] = bool(trunc[i] and not term[i])
        self.returns[done] = 0; self.lengths[done] = 0
        return obs['policy'].detach().cpu().numpy().copy(), reward, done, infos
    def close(self):
        self.env.close()
    def get_attr(self, attr_name, indices=None):
        return [getattr(self.unwrapped, attr_name) for _ in self._get_indices(indices)]
    def set_attr(self, attr_name, value, indices=None):
        if indices is not None:
            raise NotImplementedError('Per-environment config writes are unsupported')
        setattr(self.unwrapped, attr_name, value)
    def env_method(self, name, *args, indices=None, **kwargs):
        return [getattr(self.env, name)(*args, **kwargs) for _ in self._get_indices(indices)]
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
