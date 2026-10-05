"""SB3 2.9 SAC with a single actor objective: SAC + weighted demonstration loss.

Critic, entropy and Polyak updates follow the locally installed SB3 2.9 API.
Demo replay contains actual executed actions. BC uses separately recorded expert
labels at the same pre-action state, never fabricated physics transitions.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update
from common import SCHEMA, TASK, assets, fingerprint

def load_demos(path):
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta.get('schema') != SCHEMA+'-demonstrations' or meta.get('task') != TASK:
        raise ValueError('Sequence demonstrations required; legacy BC archives are not compatible')
    if meta['fingerprint'] != fingerprint() or meta['assets'] != assets():
        raise ValueError('Demonstration code/assets changed')
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
        raise ValueError('Demonstration archive hash mismatch')
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    n, dim = meta['transitions'], meta['observation_dim']
    for key, shape in [('observations',(n,dim)), ('next_observations',(n,dim)), ('actions',(n,4)),
                       ('bc_actions',(n,4)), ('rewards',(n,)), ('dones',(n,)), ('stages',(n,)), ('guidance',(n,))]:
        if key not in data or data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f'Invalid demo field {key}')
    if n < 1 or not np.isin(data['dones'], [0,1]).all() or (np.abs(data['actions']) > 1).any() or (np.abs(data['bc_actions']) > 1).any():
        raise ValueError('Invalid action/terminal bounds')
    end = 0
    for episode in meta['episodes']:
        start, stop = episode['start'], episode['stop']
        if start != end or stop <= start or stop > n or not episode['full_success']:
            raise ValueError('Only continuous fully successful physical episodes may seed replay')
        if data['dones'][stop-1] != 1 or data['dones'][start:stop-1].any():
            raise ValueError('Demo terminal flags do not match episode boundaries')
        if not np.array_equal(data['next_observations'][start:stop-1], data['observations'][start+1:stop]):
            raise ValueError('Demo transitions are discontinuous')
        end = stop
    if end != n:
        raise ValueError('Incomplete demonstration coverage')
    data['groups'] = [np.flatnonzero((data['stages'] == stage) & (data['guidance'] == level))
                      for stage in np.unique(data['stages']) for level in np.unique(data['guidance'])]
    data['groups'] = [ids for ids in data['groups'] if len(ids)]
    return data, meta

def sample_ids(data, count):
    groups = data['groups']
    return np.array([np.random.choice(groups[i]) for i in np.random.randint(len(groups), size=count)])

def bc_loss(model, data, batch_size):
    ids = sample_ids(data, batch_size)
    obs = torch.as_tensor(data['observations'][ids], device=model.device, dtype=torch.float32)
    target = torch.as_tensor(data['bc_actions'][ids], device=model.device, dtype=torch.float32).clamp(-.995,.995)
    mean, log_std, _ = model.actor.get_action_dist_params(obs)
    weights = mean.new_tensor([4.,4.,4.,1.])
    return ((mean.tanh()-target).square()*weights).mean() + .005*(log_std+4.).square().mean()

def pretrain(model, data, steps, batch_size=256):
    model.policy.set_training_mode(True)
    for step in range(steps):
        loss = bc_loss(model, data, batch_size)
        model.actor.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 5.)
        model.actor.optimizer.step()
        if (step+1)%500 == 0 or step+1 == steps:
            print(f'[BC] {step+1}/{steps} loss={float(loss.detach()):.6f}', flush=True)

class DemoReplay(ReplayBuffer):
    def __init__(self, *args, demo_path, demo_fraction=.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_path, self.demo_fraction = str(demo_path), demo_fraction
        self._data = None
    def sample(self, batch_size, env=None):
        if self._data is None:
            self._data, _ = load_demos(self.demo_path)
        count = batch_size if self.size() == 0 else max(1, round(batch_size*self.demo_fraction))
        ids = sample_ids(self._data, count)
        d = self._data
        demo = ReplayBufferSamples(self.to_torch(self._normalize_obs(d['observations'][ids],env)),
            self.to_torch(d['actions'][ids]), self.to_torch(self._normalize_obs(d['next_observations'][ids],env)),
            self.to_torch(d['dones'][ids,None]), self.to_torch(self._normalize_reward(d['rewards'][ids,None],env)))
        if count == batch_size:
            return demo
        online = super().sample(batch_size-count,env)
        return ReplayBufferSamples(*(None if a is None and b is None else torch.cat((a,b)) for a,b in zip(online,demo)))

class AnchoredSAC(SAC):
    def _excluded_save_params(self):
        return super()._excluded_save_params()+['_demo_data']

    def train(self, gradient_steps, batch_size=64):
        if not hasattr(self, '_demo_data'):
            self._demo_data, _ = load_demos(self.replay_buffer.demo_path)
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer,self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)
        logs = dict(actor=[],critic=[],bc=[],entropy=[])
        for step in range(gradient_steps):
            replay = self.replay_buffer.sample(batch_size,env=self._vec_normalize_env)
            discount = replay.discounts if replay.discounts is not None else self.gamma
            if self.use_sde:
                self.actor.reset_noise()
            actions, log_prob = self.actor.action_log_prob(replay.observations)
            log_prob = log_prob.reshape(-1,1)
            if self.ent_coef_optimizer is not None:
                coef = self.log_ent_coef.detach().exp()
                entropy_loss = -(self.log_ent_coef*(log_prob+self.target_entropy).detach()).mean()
                self.ent_coef_optimizer.zero_grad(); entropy_loss.backward(); self.ent_coef_optimizer.step()
            else:
                coef = self.ent_coef_tensor
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay.next_observations)
                next_q = torch.cat(self.critic_target(replay.next_observations,next_actions),dim=1).min(dim=1,keepdim=True).values
                target = replay.rewards + (1-replay.dones)*discount*(next_q-coef*next_log_prob.reshape(-1,1))
            critic_loss = .5*sum(F.mse_loss(q,target) for q in self.critic(replay.observations,replay.actions))
            self.critic.optimizer.zero_grad(); critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(),10.)
            self.critic.optimizer.step()
            # Freeze critic weights while retaining action gradients into the actor.
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            q = torch.cat(self.critic(replay.observations,actions),dim=1).min(dim=1,keepdim=True).values
            supervised = bc_loss(self,self._demo_data,batch_size)
            actor_loss = (coef*log_prob-q).mean() + self.bc_weight*supervised
            self.actor.optimizer.zero_grad(); actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(),5.)
            self.actor.optimizer.step()
            for parameter in self.critic.parameters():
                parameter.requires_grad_(True)
            if step%self.target_update_interval == 0:
                polyak_update(self.critic.parameters(),self.critic_target.parameters(),self.tau)
                polyak_update(self.batch_norm_stats,self.batch_norm_stats_target,1.)
            for name,value in (('actor',actor_loss),('critic',critic_loss),('bc',supervised),('entropy',coef)):
                logs[name].append(float(value.detach()))
        self._n_updates += gradient_steps
        self.logger.record('train/n_updates',self._n_updates,exclude='tensorboard')
        for name,values in logs.items():
            self.logger.record('train/'+name+'_loss' if name != 'entropy' else 'train/ent_coef',np.mean(values))
