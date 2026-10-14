"""SB3 SAC with CUDA replay, neutral initialization and residual anchoring."""
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BoundedFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, observation_space.shape[0])
    def forward(self, obs): return obs.clamp(-20, 20)


class CUDAReplay(BaseBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device='cuda:0', n_envs=1,
                 optimize_memory_usage=False, **kwargs):
        if not str(device).startswith('cuda'): raise ValueError('CUDA replay required')
        super().__init__(max(1, buffer_size // n_envs), observation_space, action_space, device, n_envs)
        shape = (self.buffer_size, n_envs)
        self.obs = torch.empty((*shape, *self.obs_shape), device=self.device)
        self.nxt = torch.empty_like(self.obs)
        self.act = torch.empty((*shape, self.action_dim), device=self.device)
        self.rew = torch.empty((*shape, 1), device=self.device)
        self.done = torch.empty_like(self.rew)
        self.handle_timeout_termination = True
        self.optimize_memory_usage = False
    def add(self, obs, next_obs, action, reward, done, infos):
        t = lambda value: torch.as_tensor(value, dtype=torch.float32, device=self.device)
        self.obs[self.pos] = t(obs); self.nxt[self.pos] = t(next_obs); self.act[self.pos] = t(action)
        self.rew[self.pos] = t(reward).reshape(self.n_envs, 1)
        timeout = np.array([info.get('TimeLimit.truncated', False) for info in infos])
        self.done[self.pos] = t(np.asarray(done) & ~timeout).reshape(self.n_envs, 1)
        self.pos += 1
        if self.pos == self.buffer_size: self.full = True; self.pos = 0
    def sample(self, batch_size, env=None):
        if not self.size(): raise ValueError('Replay is empty')
        rows = torch.randint(self.size(), (batch_size,), device=self.device)
        envs = torch.randint(self.n_envs, (batch_size,), device=self.device)
        return ReplayBufferSamples(self.obs[rows, envs], self.act[rows, envs], self.nxt[rows, envs],
                                   self.done[rows, envs], self.rew[rows, envs])
    def _get_samples(self, batch_inds, env=None): raise NotImplementedError


class ResidualSAC(SAC):
    anchor_weight = .1
    exploration_weight = .01
    def train(self, gradient_steps, batch_size=256):
        optimizer = self.actor.optimizer
        step = optimizer.step
        def anchored_step(*args, **kwargs):
            batch = self.replay_buffer.sample(batch_size)
            mean, log_std, _ = self.actor.get_action_dist_params(batch.observations)
            anchor = self.anchor_weight * mean.tanh().square().mean()
            anchor += self.exploration_weight * (log_std + 5).square().mean()
            anchor.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
            self.logger.record('train/residual_anchor', float(anchor.detach()))
            return step(*args, **kwargs)
        optimizer.step = anchored_step
        try: return super().train(gradient_steps, batch_size)
        finally: del optimizer.step


def make_model(env, seed=61001, learning_rate=3e-5, buffer_size=100000, batch_size=512):
    model = ResidualSAC('MlpPolicy', env, device='cuda:0', seed=seed, learning_rate=learning_rate,
        buffer_size=buffer_size, batch_size=batch_size, learning_starts=0, gamma=.9995, tau=.005,
        ent_coef=1e-5, train_freq=1, gradient_steps=1, replay_buffer_class=CUDAReplay,
        policy_kwargs=dict(net_arch=[256, 256], features_extractor_class=BoundedFeatures), verbose=0)
    with torch.no_grad():
        model.actor.mu.weight.zero_(); model.actor.mu.bias.zero_()
        model.actor.log_std.weight.zero_(); model.actor.log_std.bias.fill_(-5.)
    return model


def warm_critic(model, steps=512):
    from stable_baselines3.common.logger import configure
    model.set_logger(configure(None, []))
    optimizer = model.actor.optimizer
    optimizer.step = lambda *a, **kw: None
    try:
        # Bypass the residual anchor and freeze only actor updates, not the critics.
        SAC.train(model, steps, model.batch_size)
    finally: del optimizer.step
