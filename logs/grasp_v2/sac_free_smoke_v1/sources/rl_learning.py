"""Native SB3 SAC, CUDA replay and supervised actor initialization."""
import json
import numpy as np
import torch
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from rl_core import OBS_DIM,ACTION_DIM,SCHEMA

class StandardFeatures(BaseFeaturesExtractor):
    def __init__(self,observation_space,mean,std):
        super().__init__(observation_space,OBS_DIM)
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.tensor(std,dtype=torch.float32).clamp_min(.1))
    def forward(self,obs):return ((obs-self.mean)/self.std).clamp(-10,10)

def load_data(path,device):
    with np.load(path,allow_pickle=False) as f:
        meta=json.loads(str(f['metadata']))
        if meta['schema']!=SCHEMA:raise ValueError('Incompatible free-action dataset')
        data={k:torch.as_tensor(f[k].copy(),device=device) for k in f.files if k!='metadata'}
    return data,meta

class CUDAReplayBuffer(BaseBuffer):
    def __init__(self,buffer_size,observation_space,action_space,device='cuda:0',n_envs=1,
                 optimize_memory_usage=False,demonstrations=None,demo_fraction=.25,skill='lift',**kwargs):
        super().__init__(max(1,buffer_size//n_envs),observation_space,action_space,device,n_envs)
        self.obs=torch.empty((self.buffer_size,n_envs,OBS_DIM),device=self.device)
        self.nxt=torch.empty_like(self.obs);self.act=torch.empty((self.buffer_size,n_envs,ACTION_DIM),device=self.device)
        self.rew=torch.empty((self.buffer_size,n_envs,1),device=self.device);self.done=torch.empty_like(self.rew)
        self.optimize_memory_usage=False;self.handle_timeout_termination=True;self.demo_fraction=demo_fraction
        self.demonstrations=demonstrations;self.demo=None;self.only_demo=False
        if demonstrations:self.configure_demos(demonstrations,skill)

    def configure_demos(self,path,skill):
        data,meta=load_data(path,self.device);selected=torch.zeros_like(data['unassisted'],dtype=torch.bool)
        if skill=='lift':
            for ep in meta['train_episodes']:
                ids=(data['episode']==ep).nonzero().flatten()
                event=data['lift_done'][ids].nonzero().flatten()
                if len(event):selected[ids[:int(event[0])+1]]=True
        else:
            # Assisted full traces are valid for BC only. A critic for an
            # unassisted task must never learn their fictitious holding dynamics.
            selected=(data['episode']!=meta['validation_episode'])&data['unassisted']
            selected&=data['obs'][:,-8+1]<.5  # before measured lift: verified native physics prefix
        selected&=data['unassisted']
        if not bool(selected.any()):raise ValueError('No physically compatible critic demonstrations')
        done=data['lift_done'] if skill=='lift' else data['done']
        rewards=data['lift_rewards'] if skill=='lift' else data['rewards']
        self.demo=ReplayBufferSamples(data['obs'][selected],data['actions'][selected],data['next_obs'][selected],
            done[selected,None].float(),rewards[selected,None])

    def add(self,obs,next_obs,action,reward,done,infos):
        t=lambda v:torch.as_tensor(v,device=self.device,dtype=torch.float32)
        self.obs[self.pos]=t(obs);self.nxt[self.pos]=t(next_obs);self.act[self.pos]=t(action)
        self.rew[self.pos]=t(reward).reshape(self.n_envs,1)
        mask=np.array([not i.get('TimeLimit.truncated',False) for i in infos],dtype=np.float32)
        self.done[self.pos]=t(done*mask).reshape(self.n_envs,1)
        self.pos+=1
        if self.pos==self.buffer_size:self.full=True;self.pos=0

    def sample(self,batch_size,env=None):
        count=batch_size if self.only_demo or self.size()==0 else int(batch_size*self.demo_fraction) if self.demo else 0
        pieces=[]
        if count:
            ids=torch.randint(len(self.demo.observations),(count,),device=self.device)
            pieces.append(ReplayBufferSamples(*(v[ids] for v in self.demo)))
        online=batch_size-count
        if online:
            ids=torch.randint(self.size(),(online,),device=self.device);arms=torch.randint(self.n_envs,(online,),device=self.device)
            pieces.append(ReplayBufferSamples(self.obs[ids,arms],self.act[ids,arms],self.nxt[ids,arms],self.done[ids,arms],self.rew[ids,arms]))
        if not pieces:raise ValueError('No replay data')
        return ReplayBufferSamples(*(torch.cat([p[k] for p in pieces]) for k in range(5)))
    def _get_samples(self,batch_inds,env=None):raise NotImplementedError('Use CUDA sampling')

class FreeSAC(SAC):
    bc_weight=1.
    bc_data=None
    bc_indices=None
    def _excluded_save_params(self):
        return super()._excluded_save_params()+['bc_data','bc_indices']
    def train(self,gradient_steps,batch_size=256):
        super().train(gradient_steps,batch_size)
        if self.bc_data is not None and self.bc_weight>0:
            for _ in range(gradient_steps):
                ids=self.bc_indices[torch.randint(len(self.bc_indices),(batch_size,),device=self.device)]
                loss=bc_loss(self.actor,self.bc_data,ids)
                self.actor.optimizer.zero_grad();(loss*self.bc_weight).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),1.)
                self.actor.optimizer.step()
            self.logger.record('train/auxiliary_bc_loss',float(loss.detach()))

def bc_loss(actor,data,ids,std_weight=0.):
    mean,log_std,_=actor.get_action_dist_params(data['obs'][ids])
    weight=torch.tensor([100.,100.,100.,25.,25.,25.,1.]*2,device=mean.device)
    mask=data['bc_valid'][ids]*weight
    loss=(((mean.tanh()-data['actions'][ids]).square()*mask).sum()/mask.sum().clamp_min(1))
    target=torch.tensor([-6.,-6.,-6.,-6.,-6.,-6.,-4.]*2,device=mean.device)
    return loss+std_weight*(log_std-target).square().mean()

def initialize_actor(model,data,meta,updates=6000,batch_size=1024):
    ids=(data['episode']!=meta['validation_episode']).nonzero().flatten()
    # Equal physical-progress bins, not teacher stages, retain sparse jaw actions.
    bins=[]
    for b in (data['obs'][ids,-8:-4].sum(-1).long().clamp(max=4)) .unique():
        bins.append(ids[data['obs'][ids,-8:-4].sum(-1).long().clamp(max=4)==b])
    for update in range(1,updates+1):
        batch=torch.cat([x[torch.randint(len(x),(max(1,batch_size//len(bins)),),device=model.device)] for x in bins])
        loss=bc_loss(model.actor,data,batch,.002)
        model.actor.optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.actor.parameters(),1.)
        model.actor.optimizer.step()
        if update%500==0 or update==updates:print(f'[FREE SAC BC] {update}/{updates} loss={float(loss.detach()):.6f}',flush=True)
    model.bc_data=data;model.bc_indices=ids
    held=(data['episode']==meta['validation_episode']).nonzero().flatten()
    with torch.no_grad():validation=float(bc_loss(model.actor,data,held))
    return dict(weighted_validation_mse=validation,updates=updates,held_out_episode=meta['validation_episode'])
