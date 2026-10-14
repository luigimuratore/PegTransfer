"""Goal-conditioned BC of free signed pose/jaw actions, trained on CUDA."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

SCHEMA='peg-autonomy-bc-1'
# Exclude expert reference pose leads, which offer an easy autoregressive
# shortcut. Keep actual physics, jaw reference, destination and measured grip.
INDICES=[i for i in range(110) if i not in list(range(34,40))+list(range(75,81))]
OBS_DIM=len(INDICES)

def physical_observation(obs):
    return obs[...,INDICES]

class Policy(nn.Module):
    def __init__(self,mean,std):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32).clamp_min(.05))
        self.net=nn.Sequential(nn.Linear(OBS_DIM,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,14))
    def forward(self,obs):return self.net(((obs-self.mean)/self.std).clamp(-10,10)).tanh()

class NeighborPolicy(nn.Module):
    """Nonparametric BC baseline; queries physical states, never a clock/stage.

    This is a separate imitation baseline, not a neural network or fallback
    to an expert during inference. Only training episodes enter its memory.
    """
    def __init__(self,mean,std,observations,actions):
        super().__init__()
        self.register_buffer('mean',mean);self.register_buffer('std',std.clamp_min(.05))
        self.register_buffer('observations',observations);self.register_buffer('actions',actions)
        self.register_buffer('norm',observations.square().sum(-1))
    def forward(self,obs):
        x=((obs-self.mean)/self.std).clamp(-10,10)
        distance=(x.square().sum(-1,keepdim=True)+self.norm[None]-2*x@self.observations.T).clamp_min(0)
        d,ids=distance.topk(min(5,len(self.actions)),largest=False,dim=-1)
        weight=torch.exp(-(d-d[:,:1])/.02);weight=weight/weight.sum(-1,keepdim=True)
        return (self.actions[ids]*weight[...,None]).sum(-2).clamp(-1,1)

def load_policy(path,device):
    state=torch.load(path,map_location=device,weights_only=True)
    if state['schema']!=SCHEMA:raise ValueError('Incompatible BC checkpoint')
    saved=state['state_dict']
    net=(NeighborPolicy(saved['mean'],saved['std'],saved['observations'],saved['actions'])
        if state.get('architecture')=='neighbors' else Policy(saved['mean'],saved['std'])).to(device)
    net.load_state_dict(state['state_dict']);return net.eval()

def train(paths,output,device='cuda:0',updates=6000,batch_size=1024,seed=42,architecture='mlp'):
    torch.set_num_threads(8);torch.manual_seed(seed)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    all_data=[];metas=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']))
            if meta['schema']!=SCHEMA:raise ValueError('Wrong dataset schema')
            metas.append(meta)
            all_data.append({k:data[k].copy() for k in ('obs','actions','stage','validation','episode')})
    data={k:torch.as_tensor(np.concatenate([d[k] for d in all_data]),device=device) for k in all_data[0]}
    train_ids=(~data['validation'].bool()).nonzero().flatten();valid_ids=data['validation'].bool().nonzero().flatten()
    if not len(train_ids) or not len(valid_ids):raise ValueError('Separate whole validation episodes required')
    mean=data['obs'][train_ids].mean(0);std=data['obs'][train_ids].std(0).clamp_min(.05)
    if architecture=='neighbors':
        x=((data['obs'][train_ids]-mean)/std).clamp(-10,10)
        net=NeighborPolicy(mean,std,x,data['actions'][train_ids]).to(device).eval()
        with torch.inference_mode():
            errors=[]
            for ids in valid_ids.split(128):errors.append((net(data['obs'][ids])-data['actions'][ids]).square().mean())
        torch.save(dict(schema=SCHEMA,architecture=architecture,state_dict=net.state_dict()),output/'policy.pt')
        (output/'training.json').write_text(json.dumps(dict(algorithm='Nonparametric Behavior Cloning',
            architecture='5 nearest training states, physical features only',datasets=[str(p) for p in paths],
            training_transitions=len(train_ids),validation_transitions=len(valid_ids),
            validation_mse=float(torch.stack(errors).mean()),device=device,gpu=torch.cuda.get_device_name(),
            teacher_at_inference=False,physical_evaluation_required=True),indent=2)+'\n')
        return net
    net=Policy(mean,std).to(device)
    opt=torch.optim.Adam(net.parameters(),lr=1e-4)
    groups=[train_ids[data['stage'][train_ids]==stage] for stage in data['stage'][train_ids].unique()]
    weight=torch.tensor([5.,5.,5.,1.,1.,1.,3.,5.,5.,5.,1.,1.,1.,3.],device=device)
    history=[]
    for update in range(1,updates+1):
        ids=torch.cat([g[torch.randint(len(g),(max(1,batch_size//len(groups)),),device=device)] for g in groups])
        loss=((net(data['obs'][ids])-data['actions'][ids]).square()*weight).mean()
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if update%500==0 or update==updates:
            with torch.no_grad():val=float(((net(data['obs'][valid_ids])-data['actions'][valid_ids]).square()*weight).mean())
            history.append(dict(update=update,train_loss=float(loss.detach()),validation_loss=val))
            print(f'[AUTONOMY BC] {update}/{updates} train={float(loss.detach()):.7f} validation={val:.7f}',flush=True)
    torch.save(dict(schema=SCHEMA,state_dict=net.state_dict()),output/'policy.pt')
    (output/'training.json').write_text(json.dumps(dict(algorithm='Behavior Cloning',datasets=[str(p) for p in paths],
        dataset_metadata=metas,updates=updates,batch_size=batch_size,learning_rate=1e-4,device=device,
        gpu=torch.cuda.get_device_name(),architecture=[256,256],observation_dim=OBS_DIM,action_dim=14,
        teacher_inputs_at_inference=False,history=history,physical_evaluation_required=True),indent=2)+'\n')
    return net.eval()
