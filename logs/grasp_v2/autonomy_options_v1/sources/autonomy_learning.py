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

def future_actions(data,steps):
    """Supervise future measured tool poses within each COMPLETE episode.

    Larger, meaningful displacements replace tiny expert tracking errors.
    The actor still chooses all motor directions and jaws at inference.
    Terminal boundaries are clamped, never crossed into another reset.
    """
    from rl_core import rotvec,multiply,conjugate
    labels=data['actions'].clone()
    for episode in data['episode'].unique():
        ids=(data['episode']==episode).nonzero().flatten()
        future=ids[(torch.arange(len(ids),device=ids.device)+steps).clamp(max=len(ids)-1)]
        for arm in (0,1):
            base=arm*41
            xyz=[INDICES.index(base+i) for i in range(3)]
            quat=[INDICES.index(base+i) for i in range(3,7)]
            labels[ids,arm*7:arm*7+3]=(data['obs'][future][:,xyz]-data['obs'][ids][:,xyz])*.1/.003
            labels[ids,arm*7+3:arm*7+6]=rotvec(multiply(data['obs'][future][:,quat],conjugate(data['obs'][ids][:,quat])))/.15
            labels[ids,arm*7+6]=data['actions'][future,arm*7+6]
    return labels.clamp(-.999,.999)

class Policy(nn.Module):
    def __init__(self,mean,std):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32).clamp_min(.05))
        self.net=nn.Sequential(nn.Linear(OBS_DIM,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,14))
    def forward(self,obs):return self.net(((obs-self.mean)/self.std).clamp(-10,10)).tanh()

class PosePolicy(nn.Module):
    """The network predicts BOTH tool goals; generic IK follows those goals.

    No programmed task goal, phase or inactive-arm controller is supplied.
    Predicting absolute goals avoids integrating tiny output biases forever.
    """
    def __init__(self,mean,std,input_mask=None):
        super().__init__()
        self.register_buffer('mean',mean);self.register_buffer('std',std.clamp_min(.05))
        self.register_buffer('input_mask',torch.ones_like(mean) if input_mask is None else input_mask)
        self.net=nn.Sequential(nn.Linear(OBS_DIM,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,16))
    def poses(self,obs):return self.net(((obs-self.mean)/self.std).clamp(-10,10)*self.input_mask)
    def forward(self,obs):
        from rl_core import canonical,rotvec,multiply,conjugate
        poses=self.poses(obs).reshape(-1,2,8);actions=[]
        for arm in (0,1):
            xyz=[INDICES.index(arm*41+i) for i in range(3)]
            quat=[INDICES.index(arm*41+i) for i in range(3,7)]
            q=canonical(poses[:,arm,3:7]/poses[:,arm,3:7].norm(dim=-1,keepdim=True).clamp_min(1e-6))
            actions.append(torch.cat(((poses[:,arm,:3]-obs[:,xyz])*.1/.003,
                rotvec(multiply(q,conjugate(obs[:,quat])))/.15,poses[:,arm,7:8]),-1))
        return torch.cat(actions,-1).clamp(-1,1)

class OptionPosePolicy(PosePolicy):
    """A learned gate selects a learned motor head from PHYSICAL inputs.

    Options are supervised during BC only. No programmed sequencer or stage
    input is used to select an option during execution.
    """
    def __init__(self,mean,std,input_mask=None,options=6):
        super().__init__(mean,std,input_mask)
        del self.net
        self.encode=nn.Sequential(nn.Linear(OBS_DIM,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU())
        self.gate=nn.Linear(256,options);self.heads=nn.Linear(256,options*16)
        self.options=options
    def outputs(self,obs):
        x=self.encode(((obs-self.mean)/self.std).clamp(-10,10)*self.input_mask)
        return self.gate(x),self.heads(x).reshape(-1,self.options,16)
    def poses(self,obs):
        logits,heads=self.outputs(obs)
        return heads[torch.arange(len(obs),device=obs.device),logits.argmax(-1)]

def future_poses(data,steps,phase_endpoint=False):
    goals=torch.empty(len(data['obs']),16,device=data['obs'].device)
    for episode in data['episode'].unique():
        ids=(data['episode']==episode).nonzero().flatten()
        future=ids[(torch.arange(len(ids),device=ids.device)+steps).clamp(max=len(ids)-1)]
        endpoint=future.clone()
        if phase_endpoint:
            # These stages label demonstrations ONLY. The deployed actor
            # receives no stage; it must infer its goals from measured state.
            for stage in data['stage'][ids].unique():
                last=ids[data['stage'][ids]==stage][-1]
                endpoint=torch.where(data['stage'][future]==stage,last,endpoint)
        for arm in (0,1):
            pose=[INDICES.index(arm*41+i) for i in range(7)]
            goals[ids,arm*8:arm*8+7]=data['obs'][endpoint][:,pose]
            goals[ids,arm*8+7]=data['actions'][future,arm*7+6]
    return goals

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

class MemoryPolicy(nn.Module):
    """Learned physical observation history; no expert stage or clock input."""
    def __init__(self,mean,std):
        super().__init__()
        # This workstation's cuDNN recurrent backend cannot initialize with
        # the installed CUDA stack. Native ATen GRU kernels still run on CUDA.
        torch.backends.cudnn.enabled=False
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32).clamp_min(.05))
        self.encode=nn.Sequential(nn.Linear(OBS_DIM,128),nn.SiLU())
        self.memory=nn.GRU(128,128,batch_first=True)
        self.head=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,14))
        self.hidden=None
    def reset(self,ids=None):
        if ids is None:self.hidden=None
        elif self.hidden is not None:self.hidden[:,ids]=0
    def forward(self,obs):
        sequential=obs.ndim==3
        x=((obs-self.mean)/self.std).clamp(-10,10)
        x=self.encode(x if sequential else x[:,None])
        hidden=None if sequential else self.hidden
        output,hidden=self.memory(x,hidden)
        if not sequential:self.hidden=hidden.detach()
        result=self.head(output).tanh()
        return result if sequential else result[:,0]

def load_policy(path,device):
    state=torch.load(path,map_location=device,weights_only=True)
    if state['schema']!=SCHEMA:raise ValueError('Incompatible BC checkpoint')
    saved=state['state_dict']
    architecture=state.get('architecture')
    net=(NeighborPolicy(saved['mean'],saved['std'],saved['observations'],saved['actions'])
        if architecture=='neighbors' else MemoryPolicy(saved['mean'],saved['std'])
        if architecture=='gru' else OptionPosePolicy(saved['mean'],saved['std'],saved['input_mask'],len(saved['gate.bias']))
        if architecture=='mixture' else PosePolicy(saved['mean'],saved['std'])
        if architecture=='pose' else Policy(saved['mean'],saved['std'])).to(device)
    if architecture=='pose' and 'input_mask' not in saved:saved['input_mask']=torch.ones_like(saved['mean'])
    net.load_state_dict(state['state_dict']);return net.eval()

def train(paths,output,device='cuda:0',updates=6000,batch_size=1024,seed=42,architecture='mlp',future_steps=0,phase_endpoint=False):
    torch.set_num_threads(8);torch.manual_seed(seed)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    all_data=[];metas=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']))
            if meta['schema']!=SCHEMA:raise ValueError('Wrong dataset schema')
            metas.append(meta)
            sample={k:data[k].copy() for k in ('obs','actions','stage','validation','episode')}
            offset=sum(int(d['episode'].max())+1 for d in all_data) if all_data else 0
            sample['episode']+=offset
            all_data.append(sample)
    data={k:torch.as_tensor(np.concatenate([d[k] for d in all_data]),device=device) for k in all_data[0]}
    pose_targets=future_poses(data,future_steps,phase_endpoint or architecture=='mixture') if architecture in ('pose','mixture') else None
    if future_steps and architecture not in ('pose','mixture'):data['actions']=future_actions(data,future_steps)
    train_ids=(~data['validation'].bool()).nonzero().flatten();valid_ids=data['validation'].bool().nonzero().flatten()
    if not len(train_ids) or not len(valid_ids):raise ValueError('Separate whole validation episodes required')
    mean=data['obs'][train_ids].mean(0);std=data['obs'][train_ids].std(0).clamp_min(.05)
    if architecture=='mixture':
        mask=(data['obs'][train_ids].std(0)>=1e-4).float()
        if all(meta['skill']=='lift' for meta in metas):mask[35:70]=0;mask[7:19]=0;mask[77:83]=0
        options=int(data['stage'].max())+1
        stages=data['stage'].clone()
        for episode in data['episode'].unique():
            ids=(data['episode']==episode).nonzero().flatten()
            future=ids[(torch.arange(len(ids),device=device)+future_steps).clamp(max=len(ids)-1)]
            stages[ids]=data['stage'][future]
        net=OptionPosePolicy(mean,std,mask,options).to(device)
        with torch.no_grad():
            net.heads.weight.zero_()
            for stage in range(options):
                ids=train_ids[stages[train_ids]==stage]
                if len(ids):net.heads.bias[stage*16:stage*16+16].copy_(pose_targets[ids].mean(0))
        opt=torch.optim.Adam(net.parameters(),lr=1e-4)
        groups=[train_ids[stages[train_ids]==stage] for stage in range(options)]
        groups=[g for g in groups if len(g)]
        weight=torch.tensor([100.,100.,100.,10.,10.,10.,10.,3.]*2,device=device)
        history=[]
        for update in range(1,updates+1):
            ids=torch.cat([g[torch.randint(len(g),(max(1,batch_size//len(groups)),),device=device)] for g in groups])
            logits,heads=net.outputs(data['obs'][ids])
            pose=heads[torch.arange(len(ids),device=device),stages[ids]]
            loss=((pose-pose_targets[ids]).square()*weight).mean()+.5*nn.functional.cross_entropy(logits,stages[ids])
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            if update%500==0 or update==updates:
                with torch.no_grad():
                    logits,heads=net.outputs(data['obs'][valid_ids]);accuracy=float((logits.argmax(-1)==stages[valid_ids]).float().mean())
                history.append(dict(update=update,loss=float(loss.detach()),validation_option_accuracy=accuracy))
                print(f'[AUTONOMY OPTIONS BC] {update}/{updates} loss={float(loss.detach()):.7f} heldout_option_accuracy={accuracy:.3f}',flush=True)
        torch.save(dict(schema=SCHEMA,architecture='mixture',state_dict=net.state_dict()),output/'policy.pt')
        (output/'training.json').write_text(json.dumps(dict(algorithm='Hierarchical Behavior Cloning',options=options,
            learned_option_gate=True,teacher_stage_at_inference=False,datasets=[str(p) for p in paths],
            future_pose_steps=future_steps,pose_supervision='measured phase endpoints',updates=updates,batch_size=batch_size,
            learning_rate=1e-4,device=device,gpu=torch.cuda.get_device_name(),input_mask=mask.cpu().tolist(),
            physical_evaluation_required=True,history=history),indent=2)+'\n')
        return net.eval()
    if architecture=='pose':
        mask=(data['obs'][train_ids].std(0)>=1e-4).float()
        if all(meta['skill']=='lift' for meta in metas):
            mask[35:70]=0
            mask[7:19]=0  # Geometric goals do not need joint position/velocity.
            mask[77:83]=0 # Keep peg pose and forces, exclude motion-correlated shortcuts.
        net=PosePolicy(mean,std,mask).to(device)
        # Initialize inactive-arm outputs at their measured training mean.
        # The values are learned data, not a programmed hold pose; all head
        # parameters remain trainable. This avoids random outputs on constants.
        with torch.no_grad():
            net.net[-1].bias.copy_(pose_targets[train_ids].mean(0))
            net.net[-1].weight.zero_()
        opt=torch.optim.Adam(net.parameters(),lr=1e-4)
        groups=[train_ids[data['stage'][train_ids]==stage] for stage in data['stage'][train_ids].unique()]
        weight=torch.tensor([100.,100.,100.,10.,10.,10.,10.,3.]*2,device=device)
        history=[]
        for update in range(1,updates+1):
            ids=torch.cat([g[torch.randint(len(g),(max(1,batch_size//len(groups)),),device=device)] for g in groups])
            loss=((net.poses(data['obs'][ids])-pose_targets[ids]).square()*weight).mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            if update%500==0 or update==updates:
                with torch.no_grad():val=float(((net.poses(data['obs'][valid_ids])-pose_targets[valid_ids]).square()*weight).mean())
                history.append(dict(update=update,train_loss=float(loss.detach()),validation_loss=val))
                print(f'[AUTONOMY POSE BC] {update}/{updates} train={float(loss.detach()):.7f} validation={val:.7f}',flush=True)
        torch.save(dict(schema=SCHEMA,architecture='pose',state_dict=net.state_dict()),output/'policy.pt')
        (output/'training.json').write_text(json.dumps(dict(algorithm='Future-pose Behavior Cloning',datasets=[str(p) for p in paths],
            future_pose_steps=future_steps,updates=updates,batch_size=batch_size,learning_rate=1e-4,
            pose_supervision='measured phase endpoints' if phase_endpoint else 'measured future poses',
            input_mask=mask.cpu().tolist(),constant_feature_threshold=1e-4,
            architecture=[256,256],observation_dim=OBS_DIM,learned_pose_and_jaw_outputs=16,
            device=device,gpu=torch.cuda.get_device_name(),teacher_at_inference=False,
            physical_evaluation_required=True,history=history),indent=2)+'\n')
        return net.eval()
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
    if architecture=='gru':
        net=MemoryPolicy(mean,std).to(device);opt=torch.optim.Adam(net.parameters(),lr=1e-4)
        # Build true chronological sequences within whole training episodes.
        # Collection interleaves vector environments; adjacent array rows are
        # NOT temporal neighbors and must never be fed to a recurrent model.
        episodes=[];starts=[]
        length=64;burn_in=16
        for ep in data['episode'][train_ids].unique():
            ids=train_ids[data['episode'][train_ids]==ep]
            if len(ids)>=length:
                episodes.append(ids);starts.append(len(ids)-length+1)
        if not episodes:raise ValueError('Need full chronological training episodes')
        pad=torch.zeros(len(episodes),max(len(e) for e in episodes),device=device,dtype=torch.long)
        for i,ids in enumerate(episodes):pad[i,:len(ids)]=ids
        starts=torch.tensor(starts,device=device);offset=torch.arange(length,device=device)
        weight=torch.tensor([5.,5.,5.,2.,2.,2.,4.,5.,5.,5.,2.,2.,2.,4.],device=device)
        history=[]
        for update in range(1,updates+1):
            e=torch.randint(len(episodes),(64,),device=device)
            start=(torch.rand(64,device=device)*starts[e]).long()
            ids=pad[e[:,None],start[:,None]+offset]
            prediction=net(data['obs'][ids])
            loss=((prediction[:,burn_in:]-data['actions'][ids][:,burn_in:]).square()*weight).mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            if update%500==0 or update==updates:
                history.append(dict(update=update,train_loss=float(loss.detach())))
                print(f'[AUTONOMY GRU] {update}/{updates} loss={float(loss.detach()):.7f}',flush=True)
        with torch.inference_mode():
            errors=[]
            for ep in data['episode'][valid_ids].unique():
                ids=valid_ids[data['episode'][valid_ids]==ep];net.reset()
                for part in ids.split(128):
                    prediction,net.hidden=net.memory(net.encode(((data['obs'][part]-mean)/std).clamp(-10,10))[None],net.hidden)
                    errors.append((net.head(prediction[0]).tanh()-data['actions'][part]).square().mean())
            net.reset()
        torch.save(dict(schema=SCHEMA,architecture='gru',state_dict=net.state_dict()),output/'policy.pt')
        (output/'training.json').write_text(json.dumps(dict(algorithm='Recurrent Behavior Cloning',architecture='128 encoder, GRU 128, 128 head',
            future_pose_steps=future_steps,
            datasets=[str(p) for p in paths],sequence_length=length,burn_in=burn_in,batch_sequences=64,
            updates=updates,learning_rate=1e-4,device=device,gpu=torch.cuda.get_device_name(),
            validation_mse=float(torch.stack(errors).mean()),teacher_at_inference=False,physical_evaluation_required=True,history=history),indent=2)+'\n')
        return net.eval()
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
        future_pose_steps=future_steps,
        dataset_metadata=metas,updates=updates,batch_size=batch_size,learning_rate=1e-4,device=device,
        gpu=torch.cuda.get_device_name(),architecture=[256,256],observation_dim=OBS_DIM,action_dim=14,
        teacher_inputs_at_inference=False,history=history,physical_evaluation_required=True),indent=2)+'\n')
    return net.eval()
