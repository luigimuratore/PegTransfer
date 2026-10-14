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

def demonstrated_transitions(paths,options,device):
    """Infer allowed option changes from whole TRAINING episodes only."""
    counts=torch.zeros(options,options,dtype=torch.long,device=device)
    starts=torch.zeros(options,dtype=torch.long,device=device)
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            if json.loads(str(data['metadata']))['schema']!=SCHEMA:raise ValueError('Wrong transition dataset schema')
            stage=torch.as_tensor(data['stage'],device=device).long()
            episode=torch.as_tensor(data['episode'],device=device)
            validation=torch.as_tensor(data['validation'],device=device).bool()
            if bool(((stage<0)|(stage>=options)).any()):raise ValueError('Transition labels incompatible with actor')
            for ep in episode[~validation].unique():
                ids=((episode==ep)&~validation).nonzero().flatten();sequence=stage[ids]
                starts[sequence[0]]+=1
                if len(sequence)>1:
                    counts+=torch.bincount(sequence[:-1]*options+sequence[1:],minlength=options*options).reshape(options,options)
    if not bool(starts.sum()):raise ValueError('No training trajectories for transitions')
    allowed=(counts>0)|torch.eye(options,device=device,dtype=torch.bool)
    return allowed,int(starts.argmax()),counts

def physical_observation(obs):
    return obs[...,INDICES]

def future_actions(data,steps):
    """Supervise future measured tool poses within each COMPLETE episode.

    Larger, meaningful displacements replace tiny expert tracking errors.
    The actor still chooses all motor directions and jaws at inference.
    Terminal boundaries are clamped, never crossed into another reset.
    """
    from rl_core import rotvec,multiply,conjugate,bounded
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
    shaped=labels.reshape(-1,2,7)
    return torch.cat((bounded(shaped[...,:3],.999),bounded(shaped[...,3:6],.999),
        shaped[...,6:7].clamp(-.999,.999)),-1).reshape(-1,14)

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
        from rl_core import canonical,rotvec,multiply,conjugate,bounded
        poses=self.poses(obs).reshape(-1,2,8);actions=[]
        for arm in (0,1):
            xyz=[INDICES.index(arm*41+i) for i in range(3)]
            quat=[INDICES.index(arm*41+i) for i in range(3,7)]
            q=canonical(poses[:,arm,3:7]/poses[:,arm,3:7].norm(dim=-1,keepdim=True).clamp_min(1e-6))
            actions.append(torch.cat((bounded((poses[:,arm,:3]-obs[:,xyz])*.1/.003,1.),
                bounded(rotvec(multiply(q,conjugate(obs[:,quat])))/.15,1.),poses[:,arm,7:8]),-1))
        return torch.cat(actions,-1).clamp(-1,1)

class JawRateHead(nn.Module):
    """Learn incremental jaw control from force measurements and actuator state."""
    def __init__(self,options,categorical=False,coupled=False):
        super().__init__();self.options=options;self.categorical=categorical;self.coupled=coupled
        classes=6 if coupled else 5
        self.net=nn.Sequential(nn.Linear(options+9+(4 if coupled else 0),64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,classes if categorical else 1))
        if categorical:self.register_buffer('rates',torch.tensor([-1.,-.02/.15,0.,.02/.15,.05/.15,1.] if coupled else [-1.,-.02/.15,0.,.05/.15,1.]))
    def forward(self,obs,option,logits=False):
        features=[]
        for arm in (0,1):
            f=obs[:,21+35*arm:27+35*arm].reshape(-1,2,3);norm=f.norm(dim=-1)
            cosine=(f[:,0]*f[:,1]).sum(-1)/norm.prod(-1).clamp_min(1e-8)
            reference=obs[:,34+35*arm]
            measured=obs[:,19+35*arm:21+35*arm].mean(-1)
            identity=obs.new_zeros(len(obs),2);identity[:,arm]=1
            own=torch.cat((norm,norm.amin(-1,keepdim=True),norm.amax(-1,keepdim=True),
                cosine[:,None],reference[:,None],measured[:,None],identity,
                nn.functional.one_hot(option,self.options).to(obs.dtype)),1)
            if self.coupled:
                other=obs[:,21+35*(1-arm):27+35*(1-arm)].reshape(-1,2,3);other_norm=other.norm(dim=-1)
                other_cos=(other[:,0]*other[:,1]).sum(-1)/other_norm.prod(-1).clamp_min(1e-8)
                own=torch.cat((own,other_norm,other_cos[:,None],obs[:,34+35*(1-arm),None]),1)
            features.append(own)
        output=self.net(torch.stack(features,1))
        if self.categorical:return output if logits else self.rates[output.argmax(-1)]
        return output.squeeze(-1).tanh()

class PegContactPoseHead(nn.Module):
    """Bounded supervised receiver pose correction from actual local contacts."""
    def __init__(self,options,geometry_inputs=False):
        super().__init__();self.options=options
        self.register_buffer('geometry_inputs',torch.tensor(geometry_inputs))
        self.net=nn.Sequential(nn.Linear(options+(11 if geometry_inputs else 8),64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,3))
    def forward(self,obs,option,local_pose):
        from rl_core import canonical,conjugate,apply
        inverse=conjugate(canonical(obs[:,73:77]))
        force=obs[:,56:62].reshape(-1,2,3)
        local_force=apply(inverse[:,None].expand(-1,2,-1),force).reshape(-1,6)
        physical=[local_force/.4]
        if bool(self.geometry_inputs):
            relative=apply(inverse,(obs[:,35:38]-obs[:,70:73])*.1)
            physical.append((relative-local_pose[:,:3])/.001)
        features=torch.cat((*physical,obs[:,54:56],nn.functional.one_hot(option,self.options).to(obs.dtype)),1)
        from rl_core import bounded
        return bounded(self.net(features.clamp(-10,10)).tanh()*.001,.001)

class LearnedOptionRouter(nn.Module):
    """Choose the next learned option from its own goal and measured state."""
    def __init__(self,options):
        super().__init__();self.options=options
        # Absolute tool/world coordinates otherwise reproduce the recorded
        # teacher waypoint rather than detect arrival at the actor's OWN goal.
        mask=torch.zeros(OBS_DIM)
        for start,end in ((19,27),(34,35),(54,62),(69,70),(90,98)):
            mask[start:end]=1
        self.register_buffer('physical_mask',mask)
        self.register_buffer('invariant_contact_features',torch.tensor(True))
        self.register_buffer('router_uses_jaw_position',torch.tensor(False))
        self.net=nn.Sequential(nn.Linear(OBS_DIM+options+12,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,options))
    def forward(self,obs,option,goal,mean,std,mask):
        from rl_core import canonical,rotvec,multiply,conjugate
        targets=goal.reshape(-1,2,8);errors=[]
        for arm in (0,1):
            start=arm*35
            q=canonical(targets[:,arm,3:7]/targets[:,arm,3:7].norm(dim=-1,keepdim=True).clamp_min(1e-6))
            errors.extend(((targets[:,arm,:3]-obs[:,start:start+3])*10.,
                rotvec(multiply(q,conjugate(obs[:,start+3:start+7])))/.2))
        physical=((obs-mean)/std).clamp(-10,10)*mask*self.physical_mask
        if bool(self.invariant_contact_features):
            # Contact strength/opposition is independent of world direction.
            # Avoid memorizing force-vector components from one teacher pose.
            physical=torch.zeros_like(obs);physical[:,90:98]=obs[:,90:98]
            for arm in (0,1):
                offset=35*arm;force=obs[:,offset+21:offset+27].reshape(-1,2,3)
                norm=force.norm(dim=-1)
                if bool(self.router_uses_jaw_position):physical[:,offset+19:offset+21]=obs[:,offset+19:offset+21]
                physical[:,offset+21]=norm[:,0]/.4;physical[:,offset+24]=norm[:,1]/.4
                physical[:,offset+22]=(force[:,0]*force[:,1]).sum(-1)/norm.prod(-1).clamp_min(1e-8)
                if bool(self.router_uses_jaw_position):physical[:,offset+34]=obs[:,offset+34]
        identity=nn.functional.one_hot(option,self.options).to(obs.dtype)
        return self.net(torch.cat((physical,identity,torch.cat(errors,-1).clamp(-10,10)),-1))

class OptionPosePolicy(PosePolicy):
    """A learned gate selects a learned motor head from PHYSICAL inputs.

    Options are supervised during BC only. No programmed sequencer or stage
    input is used to select an option during execution.
    """
    def __init__(self,mean,std,input_mask=None,options=6,linear_goals=False,jaw_rate_head=False,jaw_categorical=False,jaw_coupled=False,option_router=False,pickup_decoder=False,contact_pose_head=False,contact_geometry_inputs=False,receiver_jaw_rate_head=False):
        super().__init__(mean,std,input_mask)
        del self.net
        self.encode=nn.Sequential(nn.Linear(OBS_DIM,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU())
        self.gate=nn.Linear(256,options)
        self.linear_goals=linear_goals
        if linear_goals:
            self.register_buffer('pose_indices',torch.tensor(list(range(70,77))+list(range(84,90)),device=mean.device))
            self.pose_head=nn.Linear(13,options*14);self.jaw_head=nn.Linear(256,options*2)
        else:self.heads=nn.Linear(256,options*16)
        self.options=options
        self.register_buffer('allowed_transitions',torch.ones(options,options,dtype=torch.bool,device=mean.device))
        self.register_buffer('initial_option',torch.tensor(-1,device=mean.device))
        self.register_buffer('latch_option_poses',torch.tensor(False,device=mean.device))
        self.register_buffer('use_jaw_rate_head',torch.tensor(jaw_rate_head,device=mean.device))
        self.register_buffer('preserve_stationary_goals',torch.tensor(False,device=mean.device))
        self.register_buffer('stationary_position',torch.zeros(options,2,dtype=torch.bool,device=mean.device))
        self.register_buffer('stationary_orientation',torch.zeros(options,2,dtype=torch.bool,device=mean.device))
        self.register_buffer('use_pickup_gate',torch.tensor(False,device=mean.device))
        self.register_buffer('track_stationary_adjustments',torch.tensor(False,device=mean.device))
        self.register_buffer('adjust_stationary_position',torch.zeros(options,2,dtype=torch.bool,device=mean.device))
        self.register_buffer('adjust_stationary_orientation',torch.zeros(options,2,dtype=torch.bool,device=mean.device))
        self.register_buffer('relative_pose_mask',torch.zeros(options,2,dtype=torch.bool,device=mean.device))
        relative=torch.zeros(options,2,7,device=mean.device);relative[...,3]=1
        self.relative_pose_goals=nn.Parameter(relative,requires_grad=False)
        self.register_buffer('target_object_mask',torch.zeros(options,2,dtype=torch.bool,device=mean.device))
        object_goals=torch.zeros(options,7,device=mean.device);object_goals[:,3]=1
        self.target_object_goals=nn.Parameter(object_goals,requires_grad=False)
        self.register_buffer('contact_pose_mask',torch.zeros(options,dtype=torch.bool,device=mean.device))
        self.contact_pose_head=PegContactPoseHead(options,contact_geometry_inputs).to(mean.device) if contact_pose_head else None
        self.jaw_rate_head=JawRateHead(options,jaw_categorical,jaw_coupled).to(mean.device) if jaw_rate_head else None
        self.receiver_jaw_rate_head=JawRateHead(options,True,True).to(mean.device) if receiver_jaw_rate_head else None
        if receiver_jaw_rate_head:
            self.register_buffer('receiver_jaw_options',torch.zeros(options,dtype=torch.bool,device=mean.device))
        self.option_router=LearnedOptionRouter(options).to(mean.device) if option_router else None
        # An embedded, frozen neural pickup decoder can preserve its measured
        # grasp behavior while the full actor learns the remaining options.
        # The full learned router still chooses every option at inference.
        if pickup_decoder and (options!=19 or not linear_goals):raise ValueError('Pickup composition requires a full linear option actor')
        self.pickup_decoder=(OptionPosePolicy(mean,std,options=6,linear_goals=True,
            jaw_rate_head=True,jaw_categorical=True).to(mean.device) if pickup_decoder else None)
        self.selected_option=None
        self.selected_goal=None
        self.selected_prediction=None
        self.current_goal=None
        self.decision_margin=.75
    def reset(self,ids=None):
        if ids is None:self.selected_option=None;self.selected_goal=None;self.selected_prediction=None;self.current_goal=None
        elif self.selected_option is not None:self.selected_option[ids]=-1
    def outputs(self,obs):
        x=self.encode(((obs-self.mean)/self.std).clamp(-10,10)*self.input_mask)
        if self.linear_goals:
            pose=((obs[:,self.pose_indices]-self.mean[self.pose_indices])/self.std[self.pose_indices]).clamp(-10,10)
            poses=self.pose_head(pose).reshape(-1,self.options,2,7)
            jaws=self.jaw_head(x).reshape(-1,self.options,2,1)
            heads=torch.cat((poses,jaws),-1).reshape(-1,self.options,16)
        else:heads=self.heads(x).reshape(-1,self.options,16)
        if bool(self.relative_pose_mask.any()):
            from rl_core import canonical,apply,multiply
            relative=self.relative_pose_goals[None].expand(len(obs),-1,-1,-1)
            pegq=canonical(obs[:,73:77])[:,None,None].expand(-1,self.options,2,-1)
            position=obs[:,None,None,70:73]+apply(pegq,relative[...,:3])/.1
            if self.contact_pose_head is not None:
                for option in self.contact_pose_mask.nonzero().flatten().tolist():
                    identity=torch.full((len(obs),),option,device=obs.device,dtype=torch.long)
                    offset=self.contact_pose_head(obs,identity,self.relative_pose_goals[option,1].expand(len(obs),-1))
                    position[:,option,1]+=apply(pegq[:,option,1],offset)/.1
            orientation=canonical(multiply(pegq,canonical(relative[...,3:7])))
            shaped=heads.reshape(-1,self.options,2,8)
            poses=torch.where(self.relative_pose_mask[None,:,:,None],torch.cat((position,orientation),-1),shaped[...,:7])
            heads=torch.cat((poses,shaped[...,7:8]),-1).reshape(-1,self.options,16)
        if bool(self.target_object_mask.any()):
            from rl_core import canonical,apply,multiply,conjugate
            # The fitted object destination is decoded through the measured
            # tool-to-object transform. Motor commands move only the tools;
            # no object pose is assigned and no attachment is introduced.
            pegq=canonical(obs[:,73:77]);inverse=conjugate(pegq)
            toolp=torch.stack((obs[:,:3],obs[:,35:38]),1)*.1
            toolq=torch.stack((obs[:,3:7],obs[:,38:42]),1)
            relativep=apply(inverse[:,None].expand(-1,2,-1),toolp-obs[:,None,70:73]*.1)
            relativeq=multiply(inverse[:,None].expand(-1,2,-1),toolq)
            desiredq=canonical(self.target_object_goals[:,3:7])[None,:,None].expand(len(obs),-1,2,-1)
            desiredp=obs[:,None,None,84:87]+self.target_object_goals[None,:,None,:3]/.1
            position=desiredp+apply(desiredq,relativep[:,None].expand(-1,self.options,-1,-1))/.1
            orientation=canonical(multiply(desiredq,relativeq[:,None].expand(-1,self.options,-1,-1)))
            shaped=heads.reshape(-1,self.options,2,8)
            poses=torch.where(self.target_object_mask[None,:,:,None],torch.cat((position,orientation),-1),shaped[...,:7])
            heads=torch.cat((poses,shaped[...,7:8]),-1).reshape(-1,self.options,16)
        if self.pickup_decoder is not None:
            _,pickup_heads=self.pickup_decoder.outputs(obs)
            heads=torch.cat((pickup_heads,heads[:,6:]),1)
        return self.gate(x),heads
    def poses(self,obs):
        logits,heads=self.outputs(obs)
        rows=torch.arange(len(obs),device=obs.device)
        if self.selected_option is None or len(self.selected_option)!=len(obs):
            self.selected_option=torch.full((len(obs),),-1,device=obs.device,dtype=torch.long)
        uninitialized=self.selected_option<0
        previous=self.selected_option.clamp_min(0)
        if self.option_router is not None:
            context_goal=heads[rows,previous]
            if self.selected_goal is not None and len(self.selected_goal)==len(obs):
                prior_goal=self.current_goal if self.current_goal is not None and bool(self.track_stationary_adjustments) else self.selected_goal
                context_goal=torch.where(uninitialized[:,None],context_goal,prior_goal)
            logits=self.option_router(obs,previous,context_goal,self.mean,self.std,self.input_mask)
        allowed=self.allowed_transitions
        if self.pickup_decoder is not None and bool(self.use_pickup_gate):
            pickup_logits,_=self.pickup_decoder.outputs(obs)
            if hasattr(self,'pickup_boundary'):
                scores=self.pickup_boundary(pickup_boundary_features(obs))
                boundary=torch.full_like(pickup_logits,-torch.inf)
                boundary[:,3:5]=scores
                pickup_logits=torch.where((previous==3)[:,None],boundary,pickup_logits)
            pickup_scores=logits.new_full(logits.shape,-torch.inf)
            pickup_scores[:,:6]=pickup_logits
            # The pickup network retains its own learned decisions. At its
            # final learned option, the full router may choose an option in
            # the rest of its learned repertoire. No contact threshold or
            # elapsed-time rule makes this choice.
            full_best=logits.argmax(-1)
            full_change=(previous==5)&(full_best>=6)&self.allowed_transitions[previous,full_best]&(
                logits[rows,full_best]-logits[rows,previous]>self.decision_margin)
            use_pickup=(previous<6)&~full_change
            logits=torch.where(use_pickup[:,None],pickup_scores,logits)
            allowed=allowed.clone();allowed[:6,:6]=self.pickup_decoder.allowed_transitions
        best=logits.argmax(-1)
        self.decision_scores=logits.detach()
        # Generic decision hysteresis: retain the network's own last choice
        # until a different learned head has a clear logit advantage. There is
        # no prescribed option order, task transition, teacher phase or clock.
        switch=uninitialized|(allowed[previous,best]&(logits[rows,best]-logits[rows,previous]>self.decision_margin))
        old_option=self.selected_option.clone()
        self.selected_option=torch.where(switch,best,self.selected_option)
        if int(self.initial_option)>=0:
            self.selected_option=torch.where(uninitialized,self.initial_option,self.selected_option)
        goal=heads[rows,self.selected_option]
        raw_prediction=goal
        changed=(self.selected_option!=old_option)|uninitialized
        if bool(self.track_stationary_adjustments):
            if self.selected_prediction is None or len(self.selected_prediction)!=len(obs):self.selected_prediction=raw_prediction.detach().clone()
            self.selected_prediction=torch.where(changed[:,None],raw_prediction.detach(),self.selected_prediction)
        if bool(self.preserve_stationary_goals):
            measured=torch.stack((torch.cat((obs[:,:7],obs.new_zeros(len(obs),1)),1),
                torch.cat((obs[:,35:42],obs.new_zeros(len(obs),1)),1)),1)
            anchor=measured
            if self.selected_goal is not None and len(self.selected_goal)==len(obs):
                prior_goal=self.current_goal if self.current_goal is not None and bool(self.track_stationary_adjustments) else self.selected_goal
                anchor=torch.where(uninitialized[:,None,None],measured,prior_goal.reshape(-1,2,8))
            candidate=goal.reshape(-1,2,8)
            position=torch.where(self.stationary_position[self.selected_option,:,None],anchor[...,:3],candidate[...,:3])
            orientation=torch.where(self.stationary_orientation[self.selected_option,:,None],anchor[...,3:7],candidate[...,3:7])
            goal=torch.cat((position,orientation,candidate[...,7:8]),-1).reshape(-1,16)
        if bool(self.latch_option_poses):
            if self.selected_goal is None or len(self.selected_goal)!=len(obs):self.selected_goal=goal.detach().clone()
            self.selected_goal=torch.where(changed[:,None],goal.detach(),self.selected_goal)
            cached=self.selected_goal.reshape(-1,2,8);current=goal.reshape(-1,2,8)
            goal=torch.cat((cached[...,:7],current[...,7:8]),-1).reshape(-1,16)
        if bool(self.track_stationary_adjustments):
            from rl_core import canonical,multiply,conjugate,quaternion,rotvec,bounded
            # Share the previous neural goal, then follow only the CHANGE in
            # this option's learned prediction since entry. This removes an
            # endpoint jump while retaining the learned geometric response.
            baseline=goal.reshape(-1,2,8);fresh=raw_prediction.reshape(-1,2,8)
            entry=self.selected_prediction.reshape(-1,2,8)
            position=baseline[...,:3]+bounded((fresh[...,:3]-entry[...,:3])*.1,.002)/.1
            fresh_q=canonical(fresh[...,3:7]/fresh[...,3:7].norm(dim=-1,keepdim=True).clamp_min(1e-6))
            entry_q=canonical(entry[...,3:7]/entry[...,3:7].norm(dim=-1,keepdim=True).clamp_min(1e-6))
            rotation=bounded(rotvec(multiply(fresh_q,conjugate(entry_q))),.13)
            orientation=canonical(multiply(quaternion(rotation),baseline[...,3:7]))
            position=torch.where(self.adjust_stationary_position[self.selected_option,:,None],position,baseline[...,:3])
            orientation=torch.where(self.adjust_stationary_orientation[self.selected_option,:,None],orientation,baseline[...,3:7])
            goal=torch.cat((position,orientation,baseline[...,7:8]),-1).reshape(-1,16)
        if self.jaw_rate_head is not None:
            rate=self.jaw_rates(obs,self.selected_option)
            reference=torch.stack((obs[:,34],obs[:,69]),1)*.5
            jaw=(reference+rate*.15*.01).clamp(0,.5)
            shaped=goal.reshape(-1,2,8)
            goal=torch.cat((shaped[...,:7],(4*jaw-1)[...,None]),-1).reshape(-1,16)
        self.current_goal=goal.detach()
        return goal

    def jaw_rates(self,obs,option):
        """Neural feedback; isolate receiver recovery from the preserved donor.

        The optional head is trained on recorded receiver jaw actions. Its mask
        only chooses the trained head for the actor's own learned behavior;
        it adds no physical threshold, teacher, timer or transition rule.
        """
        rate=self.jaw_rate_head(obs,option)
        if self.pickup_decoder is not None:
            pickup_rate=self.pickup_decoder.jaw_rate_head(obs,option.clamp_max(5))
            rate=torch.where((option<6)[:,None],pickup_rate,rate)
        if self.receiver_jaw_rate_head is not None:
            receiver=self.receiver_jaw_rate_head(obs,option)[:,1]
            rate=rate.clone()
            rate[:,1]=torch.where(self.receiver_jaw_options[option],receiver,rate[:,1])
        return rate

def option_entry_observations(data,stages):
    """Offline labels: repeat each contiguous option's entry state in its episode.

    A latched pose is chosen at entry, so fitting it on the moving peg later
    in the option teaches a different input distribution than deployment.
    Episodes can be interleaved; repeated visits to an option remain separate.
    """
    entries=torch.empty_like(data['obs'])
    for episode in data['episode'].unique():
        ids=(data['episode']==episode).nonzero().flatten()
        local=stages[ids]
        changes=torch.cat((torch.ones(1,device=ids.device,dtype=torch.bool),local[1:]!=local[:-1]))
        starts=changes.nonzero().flatten()
        entries[ids]=data['obs'][ids[starts[changes.long().cumsum(0)-1]]]
    return entries

def completed_option_pose_mask(data,stages):
    """A reset/retry endpoint is not an achieved motor destination.

    Retain aborted visits for gate/recovery learning, but exclude their pose
    endpoints. Inputs are successful complete demonstrations; their final
    segment is therefore valid. Forward skips also represent completed visits.
    """
    valid=torch.zeros(len(stages),device=stages.device,dtype=torch.bool)
    for episode in data['episode'].unique():
        ids=(data['episode']==episode).nonzero().flatten();local=stages[ids]
        changes=torch.cat((torch.ones(1,device=ids.device,dtype=torch.bool),local[1:]!=local[:-1]))
        starts=changes.nonzero().flatten();values=local[starts]
        completed=torch.cat((values[1:]>values[:-1],torch.ones(1,device=ids.device,dtype=torch.bool)))
        valid[ids]=completed[changes.long().cumsum(0)-1]
    return valid

def demonstrated_relative_poses(paths,options,arm,selected,device):
    """Fit tool-to-peg pose templates from completed TRAINING visits.

    Structured BC decoder: observed peg transforms carry a learned local
    pose into world coordinates. Templates are fitted, not geometry-derived
    teacher destinations. Validation episodes never enter the fit.
    """
    from rl_core import canonical,conjugate,apply,multiply,rotvec
    examples={option:[] for option in selected};heldout={option:[] for option in selected}
    for path in paths:
        with np.load(path,allow_pickle=False) as archive:
            meta=json.loads(str(archive['metadata']))
            if meta['schema']!=SCHEMA or meta['skill']!='full' or meta.get('assisted_grasp') is not False or meta.get('collection_complete') is not True:
                raise ValueError('Complete native full demonstrations required')
            data={k:torch.as_tensor(archive[k],device=device) for k in ('obs','stage','episode','validation')}
        valid=completed_option_pose_mask(data,data['stage'])
        for ep in data['episode'].unique():
            ids=(data['episode']==ep).nonzero().flatten();local=data['stage'][ids].long()
            if len(data['validation'][ids].unique())!=1:raise ValueError('Whole-episode split required')
            starts=torch.cat((ids.new_zeros(1),(local[1:]!=local[:-1]).nonzero().flatten()+1))
            ends=torch.cat((starts[1:],starts.new_tensor([len(ids)])))
            for begin,end in zip(starts.tolist(),ends.tolist()):
                option=int(local[begin]);segment=ids[begin:end]
                if option not in examples or not bool(valid[segment].all()):continue
                row=data['obs'][segment[-min(5,len(segment)):]]
                inverse=conjugate(canonical(row[:,73:77]))
                position=apply(inverse,(row[:,arm*35:arm*35+3]-row[:,70:73])*.1)
                orientation=canonical(multiply(inverse,row[:,arm*35+3:arm*35+7]))
                item=torch.cat((position,orientation),-1)
                target=heldout if bool(data['validation'][ids[0]]) else examples
                target[option].append(item)
    poses=torch.zeros(options,2,7,device=device);poses[...,3]=1;mask=torch.zeros(options,2,dtype=torch.bool,device=device);evidence=[]
    for option in selected:
        if not examples[option]:raise ValueError(f'No training visits for option {option}')
        samples=torch.cat(examples[option]);q=samples[:,3:7]
        q=torch.where((q*q[:1]).sum(-1,keepdim=True)<0,-q,q)
        fitted=torch.cat((samples[:,:3].mean(0),canonical(q.mean(0))),-1)
        poses[option,arm]=fitted;mask[option,arm]=True
        metrics={}
        if heldout[option]:
            validation=torch.cat(heldout[option])
            metrics=dict(heldout_position_rmse_mm=float((validation[:,:3]-fitted[:3]).square().sum(-1).mean().sqrt()*1000),
                heldout_rotation_rmse_rad=float(rotvec(multiply(validation[:,3:7],conjugate(fitted[3:7].expand(len(validation),-1)))).square().sum(-1).mean().sqrt()))
        evidence.append(dict(option=option,arm=arm,training_visits=len(examples[option]),heldout_visits=len(heldout[option]),
            learned_local_pose=fitted.cpu().tolist(),**metrics))
    return poses,mask,evidence

def demonstrated_target_objects(paths,options,selected,device):
    """Fit object destinations relative to the target from TRAIN endpoints."""
    from rl_core import canonical,conjugate,multiply,rotvec
    examples={option:[] for option in selected};heldout={option:[] for option in selected}
    for path in paths:
        with np.load(path,allow_pickle=False) as archive:
            meta=json.loads(str(archive['metadata']))
            if meta['schema']!=SCHEMA or meta['skill']!='full' or meta.get('assisted_grasp') is not False or meta.get('collection_complete') is not True:
                raise ValueError('Complete native full demonstrations required')
            data={k:torch.as_tensor(archive[k],device=device) for k in ('obs','stage','episode','validation')}
        valid=completed_option_pose_mask(data,data['stage'])
        for ep in data['episode'].unique():
            ids=(data['episode']==ep).nonzero().flatten();local=data['stage'][ids].long()
            if len(data['validation'][ids].unique())!=1:raise ValueError('Whole-episode split required')
            starts=torch.cat((ids.new_zeros(1),(local[1:]!=local[:-1]).nonzero().flatten()+1));ends=torch.cat((starts[1:],starts.new_tensor([len(ids)])))
            for begin,end in zip(starts.tolist(),ends.tolist()):
                option=int(local[begin]);segment=ids[begin:end]
                if option not in examples or not bool(valid[segment].all()):continue
                row=data['obs'][segment[-min(5,len(segment)):]]
                item=torch.cat(((row[:,70:73]-row[:,84:87])*.1,canonical(row[:,73:77])),-1)
                target=heldout if bool(data['validation'][ids[0]]) else examples;target[option].append(item)
    goals=torch.zeros(options,7,device=device);goals[:,3]=1;mask=torch.zeros(options,2,dtype=torch.bool,device=device);evidence=[]
    for option in selected:
        if not examples[option]:raise ValueError(f'No training endpoints for option {option}')
        samples=torch.cat(examples[option]);q=samples[:,3:7];q=torch.where((q*q[:1]).sum(-1,keepdim=True)<0,-q,q)
        fitted=torch.cat((samples[:,:3].mean(0),canonical(q.mean(0))),-1)
        goals[option]=fitted;mask[option,1]=True;metrics={}
        if heldout[option]:
            validation=torch.cat(heldout[option])
            metrics=dict(heldout_position_rmse_mm=float((validation[:,:3]-fitted[:3]).square().sum(-1).mean().sqrt()*1000),
                heldout_rotation_rmse_rad=float(rotvec(multiply(validation[:,3:7],conjugate(fitted[3:7].expand(len(validation),-1)))).square().sum(-1).mean().sqrt()))
        evidence.append(dict(option=option,training_visits=len(examples[option]),heldout_visits=len(heldout[option]),learned_target_object_pose=fitted.cpu().tolist(),**metrics))
    return goals,mask,evidence

def demonstrated_stationary_goals(paths,options,device,position_tolerance=.0005,angle_tolerance=.02):
    """Infer held motor references from complete TRAINING visits only.

    Actions reconstruct tool motor references, never peg pose commands. These
    learned stationary masks share an actor's previous goal across options;
    they do not select an option or supply a task destination at inference.
    """
    from rl_core import quaternion,multiply,conjugate,rotvec,canonical
    excursion=[[] for _ in range(options)];angles=[[] for _ in range(options)]
    for path in paths:
        with np.load(path,allow_pickle=False) as archive:
            meta=json.loads(str(archive['metadata']))
            if meta['schema']!=SCHEMA or meta.get('assisted_grasp') is not False:raise ValueError('Native demonstrations required')
            data={k:torch.as_tensor(archive[k],device=device) for k in ('obs','actions','stage','episode','validation')}
        valid=completed_option_pose_mask(data,data['stage'])
        for episode in data['episode'][~data['validation'].bool()].unique():
            ids=((data['episode']==episode)&~data['validation'].bool()).nonzero().flatten()
            local=data['stage'][ids].long()
            starts=torch.cat((ids.new_zeros(1),(local[1:]!=local[:-1]).nonzero().flatten()+1))
            ends=torch.cat((starts[1:],starts.new_tensor([len(ids)])))
            for begin,end in zip(starts.tolist(),ends.tolist()):
                segment=ids[begin:end];option=int(local[begin])
                if not bool(valid[segment].all()):continue
                obs=data['obs'][segment];actions=data['actions'][segment].reshape(-1,2,7)
                p=torch.stack((obs[:,:3],obs[:,35:38]),1)*.1+actions[...,:3]*.003
                q=torch.stack((obs[:,3:7],obs[:,38:42]),1)
                q=canonical(multiply(quaternion(actions[...,3:6]*.15),q))
                excursion[option].append((p-p[:1]).norm(dim=-1).amax(0))
                angles[option].append(rotvec(multiply(q,conjugate(q[:1].expand_as(q)))).norm(dim=-1).amax(0))
    position=torch.zeros(options,2,dtype=torch.bool,device=device);orientation=position.clone();details=[]
    for option in range(options):
        if excursion[option]:
            distance=torch.quantile(torch.stack(excursion[option]),.9,dim=0)
            angle=torch.quantile(torch.stack(angles[option]),.9,dim=0)
            position[option]=distance<position_tolerance;orientation[option]=angle<angle_tolerance
            details.append(dict(option=option,training_visits=len(excursion[option]),reference_excursion_mm=(distance*1000).cpu().tolist(),
                reference_rotation_rad=angle.cpu().tolist(),hold_position=position[option].cpu().tolist(),hold_orientation=orientation[option].cpu().tolist()))
        else:details.append(dict(option=option,training_visits=0,hold_position=[False,False],hold_orientation=[False,False]))
    return position,orientation,details

def future_poses(data,steps,phase_endpoint=False):
    goals=torch.empty(len(data['obs']),16,device=data['obs'].device)
    for episode in data['episode'].unique():
        ids=(data['episode']==episode).nonzero().flatten()
        future=ids[(torch.arange(len(ids),device=ids.device)+steps).clamp(max=len(ids)-1)]
        endpoint=future.clone()
        if phase_endpoint:
            # These stages label demonstrations ONLY. The deployed actor
            # receives no stage; it must infer its goals from measured state.
            stage=data['stage'][ids]
            changes=torch.cat((torch.ones(1,device=ids.device,dtype=torch.bool),stage[1:]!=stage[:-1]))
            segment=changes.long().cumsum(0)-1
            ends=torch.cat(((stage[1:]!=stage[:-1]).nonzero().flatten(),torch.tensor([len(ids)-1],device=ids.device)))
            local_future=(torch.arange(len(ids),device=ids.device)+steps).clamp(max=len(ids)-1)
            endpoint=ids[ends[segment[local_future]]]
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

def pickup_boundary_features(obs):
    forces=obs[:,21:27].reshape(-1,2,3)
    magnitudes=forces.norm(dim=-1)
    opposition=-(forces[:,0]*forces[:,1]).sum(-1)/(magnitudes.prod(-1)+1e-6)
    return torch.cat((magnitudes.sort(-1).values,opposition[:,None],
        obs[:,19:21],obs[:,34:35],obs[:,90:91],obs[:,94:95]),-1)

def load_policy(path,device):
    state=torch.load(path,map_location=device,weights_only=True)
    if state['schema']!=SCHEMA:raise ValueError('Incompatible BC checkpoint')
    saved=state['state_dict']
    architecture=state.get('architecture')
    net=(NeighborPolicy(saved['mean'],saved['std'],saved['observations'],saved['actions'])
        if architecture=='neighbors' else MemoryPolicy(saved['mean'],saved['std'])
        if architecture=='gru' else OptionPosePolicy(saved['mean'],saved['std'],saved['input_mask'],len(saved['gate.bias']),architecture=='linear_options',bool(saved.get('use_jaw_rate_head',False)),
            'jaw_rate_head.rates' in saved,len(saved.get('jaw_rate_head.rates',[]))==6,'option_router.net.0.weight' in saved,'pickup_decoder.mean' in saved,'contact_pose_head.net.0.weight' in saved,
            'contact_pose_head.net.0.weight' in saved and saved['contact_pose_head.net.0.weight'].shape[1]==len(saved['gate.bias'])+11,
            'receiver_jaw_rate_head.rates' in saved)
        if architecture in ('mixture','linear_options') else PosePolicy(saved['mean'],saved['std'])
        if architecture=='pose' else Policy(saved['mean'],saved['std'])).to(device)
    if architecture=='pose' and 'input_mask' not in saved:saved['input_mask']=torch.ones_like(saved['mean'])
    if architecture in ('mixture','linear_options'):
        saved.setdefault('allowed_transitions',net.allowed_transitions)
        saved.setdefault('initial_option',net.initial_option)
        saved.setdefault('latch_option_poses',net.latch_option_poses)
        saved.setdefault('use_jaw_rate_head',net.use_jaw_rate_head)
        saved.setdefault('preserve_stationary_goals',net.preserve_stationary_goals)
        saved.setdefault('stationary_position',net.stationary_position)
        saved.setdefault('stationary_orientation',net.stationary_orientation)
        saved.setdefault('use_pickup_gate',net.use_pickup_gate)
        saved.setdefault('contact_pose_mask',net.contact_pose_mask)
        if net.contact_pose_head is not None:saved.setdefault('contact_pose_head.geometry_inputs',net.contact_pose_head.geometry_inputs)
        for key in ('track_stationary_adjustments','adjust_stationary_position','adjust_stationary_orientation','relative_pose_mask','relative_pose_goals','target_object_mask','target_object_goals'):
            saved.setdefault(key,getattr(net,key))
        if net.pickup_decoder is not None:
            saved.setdefault('pickup_decoder.use_pickup_gate',net.pickup_decoder.use_pickup_gate)
            saved.setdefault('pickup_decoder.contact_pose_mask',net.pickup_decoder.contact_pose_mask)
            for key in ('track_stationary_adjustments','adjust_stationary_position','adjust_stationary_orientation','relative_pose_mask','relative_pose_goals','target_object_mask','target_object_goals'):
                saved.setdefault('pickup_decoder.'+key,getattr(net.pickup_decoder,key))
        if net.option_router is not None:
            saved.setdefault('option_router.physical_mask',torch.ones_like(net.mean))
            saved.setdefault('option_router.invariant_contact_features',torch.tensor(False,device=net.mean.device))
            saved.setdefault('option_router.router_uses_jaw_position',torch.tensor(True,device=net.mean.device))
    if 'pickup_boundary.0.weight' in saved:
        net.pickup_boundary=torch.nn.Sequential(torch.nn.Linear(8,32),torch.nn.Tanh(),torch.nn.Linear(32,32),torch.nn.Tanh(),torch.nn.Linear(32,2)).to(device)
    net.load_state_dict(state['state_dict']);return net.eval()

def contact_pose_examples(paths,net,device):
    """Measured one-step expert references; keep whole validation episodes out."""
    from rl_core import canonical,conjugate,apply,bounded
    arrays=[];offset=0
    for path in paths:
        with np.load(path,allow_pickle=False) as archive:
            metadata=json.loads(str(archive['metadata']))
            if metadata['schema']!=SCHEMA or metadata['skill']!='full' or metadata.get('assisted_grasp') is not False or metadata.get('collection_complete') is not True:
                raise ValueError('Complete native full demonstrations required')
            item={k:archive[k].copy() for k in ('obs','actions','stage','episode','validation')}
        if set(np.unique(item['episode']).tolist())!=set(metadata['episodes']):raise ValueError('Episode provenance mismatch')
        for ep in np.unique(item['episode']):
            if len(np.unique(item['validation'][item['episode']==ep]))!=1:raise ValueError('Whole-episode split required')
        item['episode']+=offset;offset=int(item['episode'].max())+1;arrays.append(item)
    data={k:torch.as_tensor(np.concatenate([a[k] for a in arrays]),device=device) for k in arrays[0]}
    selected=torch.isin(data['stage'],torch.tensor([10,11],device=device))
    selected&=completed_option_pose_mask(data,data['stage'])
    data={k:v[selected] for k,v in data.items()}
    if not len(data['obs']) or not bool(torch.isfinite(data['obs']).all()) or not bool(torch.isfinite(data['actions']).all()):raise ValueError('Invalid contact examples')
    inverse=conjugate(canonical(data['obs'][:,73:77]))
    # Action is the recorded pre-action TOOL reference, not a future object pose.
    command=data['obs'][:,35:38]*.1+data['actions'][:,7:10]*.003
    reference=apply(inverse,command-data['obs'][:,70:73]*.1)
    local_pose=net.relative_pose_goals[data['stage'].long(),1]
    data['target']=bounded(reference-local_pose[:,:3],.001)
    if not bool((~data['validation'].bool()).any()) or not bool(data['validation'].bool().any()):raise ValueError('Whole held-out episodes required')
    return data

def fit_contact_pose(policy,paths,output,device,updates=6000,batch_size=1024,seed=42):
    """Fit only a small contact correction network; preserve all prior motors."""
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.manual_seed(seed);torch.set_num_threads(8)
    net=load_policy(policy,device)
    if not isinstance(net,OptionPosePolicy) or net.options!=19 or not bool(net.relative_pose_mask[10:12,1].all()) or not bool(net.track_stationary_adjustments):
        raise ValueError('Fitted relative full actor with bounded feedback required')
    for parameter in net.parameters():parameter.requires_grad_(False)
    data=contact_pose_examples(paths,net,device)
    train=(~data['validation'].bool()).nonzero().flatten();valid=data['validation'].bool().nonzero().flatten()
    net.contact_pose_head=PegContactPoseHead(net.options).to(device);net.contact_pose_mask[10:12]=True
    optimizer=torch.optim.Adam(net.contact_pose_head.parameters(),lr=1e-4);history=[]
    groups=[train[data['stage'][train]==option] for option in (10,11)]
    if any(not len(group) for group in groups):raise ValueError('Both contact visits required')
    def prediction(ids):
        option=data['stage'][ids].long()
        return net.contact_pose_head(data['obs'][ids],option,net.relative_pose_goals[option,1])
    for update in range(1,updates+1):
        ids=torch.cat([group[torch.randint(len(group),(max(1,batch_size//2),),device=device)] for group in groups])
        loss=((prediction(ids)-data['target'][ids])/.001).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        if update%500==0 or update==updates:
            with torch.no_grad():
                error=float((prediction(valid)-data['target'][valid]).square().sum(-1).mean().sqrt()*1000)
            history.append(dict(update=update,loss=float(loss.detach()),heldout_pose_rmse_mm=error))
            print(f'[AUTONOMY CONTACT BC] {update}/{updates} loss={float(loss.detach()):.6f} heldout_mm={error:.4f}',flush=True)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    state=torch.load(policy,map_location=device,weights_only=True);state['state_dict']=net.state_dict();torch.save(state,output/'policy.pt')
    config=json.loads((Path(policy).parent/'training.json').read_text())
    config.update(retrained=True,parent_policy=str(policy),physical_evaluation_required=True,
        implementation_change='Learn bounded receiver local contact corrections from recorded pre-action native tool references',
        contact_pose_updates=updates,contact_pose_learning_rate=1e-4,contact_pose_batch_size=batch_size,
        contact_pose_datasets=[str(path) for path in paths],contact_pose_history=history,contact_pose_bound_m=.001,
        contact_pose_training_episodes=int(data['episode'][train].unique().numel()),contact_pose_validation_episodes=int(data['episode'][valid].unique().numel()),
        movement_router_and_jaw_parameters_frozen=True,contact_pose_inputs='Actual peg-frame finger forces and jaws, own selected behavior; no tool position/teacher/clock/reward',
        contact_pose_algorithm='Supervised BC contact correction; not SAC')
    (output/'training.json').write_text(json.dumps(config,indent=2)+'\n')
    return net.eval()

def teacher_pressure_rates(obs,actions,stage,minimum=.30,maximum=.40):
    """Offline labels reconstruct the native pickup expert's pressure decision.

    Used ONLY for training, never by the actor or simulation controller.
    The earlier absolute jaw target also contained actuator tracking lag.
    """
    reference=obs[:,[34,69]]*.5
    rates=(((actions[:,[6,13]]+1)*.25-reference)/(.15*.01)).clamp(-1,1)
    force=obs[:,21:27].reshape(-1,2,3).norm(dim=-1)
    pressure=torch.where(force.amin(-1)<minimum,torch.where(force.amax(-1)>.03,-.02/.15,-1.),
        torch.where(force.amin(-1)>maximum,.05/.15,0.))
    rates[:,0]=torch.where(stage>=3,pressure,rates[:,0])
    return rates

def full_teacher_pressure_rates(obs,actions,stage,minimum=.36,maximum=.40):
    """Offline full-demo labels, including load transfer; NEVER an actor rule."""
    rates=teacher_pressure_rates(obs,actions,stage,minimum,maximum)
    reference=obs[:,[34,69]]*.5
    recorded=(((actions[:,[6,13]]+1)*.25-reference)/(.15*.01)).clamp(-1,1)
    forces=obs[:,[21,22,23,24,25,26,56,57,58,59,60,61]].reshape(-1,2,2,3)
    norm=forces.norm(dim=-1);receiver=norm[:,1]
    cosine=(forces[:,1,0]*forces[:,1,1]).sum(-1)/receiver.prod(-1).clamp_min(1e-8)
    supported=(receiver.amin(-1)>=.36)&(cosine<-.25)
    rates[:,0]=torch.where(stage==11,torch.where(supported,.02/.15,-.02/.15),rates[:,0])
    rates[:,0]=torch.where(stage>=12,recorded[:,0],rates[:,0])
    pressure=torch.where(receiver.amin(-1)<.36,-.02/.15,torch.where(receiver.amin(-1)>.42,.05/.15,0.))
    imbalance=(receiver[:,1]-receiver[:,0])/receiver.sum(-1).clamp_min(1e-8)
    balancing=(stage==10)&(receiver.amax(-1)>.10)&(imbalance.abs()>.2)
    pressure=torch.where(balancing,0.,pressure)
    rates[:,1]=torch.where((stage>=10)&(stage<=15),pressure,recorded[:,1])
    return rates

def calibrate_opening_observations(obs,stage,references,full=False):
    """Offline opening examples; preserve the donor's loaded contact state."""
    calibrated=obs.clone()
    donor=stage<3
    calibrated[donor,34]=references[donor,0]/.5
    calibrated[donor,19:21]=references[donor,0,None]/.5
    receiver=stage<10 if full else donor
    calibrated[receiver,69]=references[receiver,1]/.5
    calibrated[receiver,54:56]=references[receiver,1,None]/.5
    return calibrated

def causal_router_examples(data):
    """Stay labels and observed changes, conditioned on the PREVIOUS goal.

    Only measurements at/after a demonstrated transition can label a change.
    No future flags, elapsed time or teacher stages are supplied at runtime.
    Keep visits separate, including retries and asynchronously interleaved IDs.
    """
    observations=[];contexts=[];entries=[];labels=[]
    stage=data['stage'].long()
    for episode in data['episode'].unique():
        ids=(data['episode']==episode).nonzero().flatten();local=stage[ids]
        starts=torch.cat((torch.zeros(1,device=ids.device,dtype=torch.long),
            (local[1:]!=local[:-1]).nonzero().flatten()+1))
        ends=torch.cat((starts[1:],starts.new_tensor([len(ids)])))
        for visit,(begin,end) in enumerate(zip(starts.tolist(),ends.tolist())):
            current=ids[begin:end]
            observations.append(current);contexts.append(stage[current]);labels.append(stage[current])
            entries.append(ids[begin].expand(len(current)))
            if visit:
                # Positive examples use the OLD cached goal with the NEW
                # measured state. Duplicate at most 20 real samples after the
                # change, not samples before contact/hold was actually valid.
                changed=current[:20];previous=ids[starts[visit-1]]
                observations.append(changed);contexts.append(stage[previous].expand(len(changed)))
                labels.append(stage[changed]);entries.append(previous.expand(len(changed)))
    return dict(indices=torch.cat(observations),context=torch.cat(contexts),
        entry=torch.cat(entries),label=torch.cat(labels))

def calibrate_router_context(observations,targets,context,label):
    """Offline internal-goal/force calibration; never a physical success record.

    Retain observed contact/progress flags. Vary pressure only on already
    observed successful close transitions, covering the jaw actor's band.
    This function is used exclusively in training, not policy inference.
    """
    observations=observations.clone();targets=targets.clone();changed=label!=context
    for arm in (0,1):
        targets[changed,arm*8:arm*8+7]=observations[changed,arm*35:arm*35+7]
        closing=changed&(context==(3 if arm==0 else 10))
        if bool(closing.any()):
            targets[closing,arm*8:arm*8+3]+=(torch.rand(int(closing.sum()),3,device=targets.device)*2-1)*.003
            f=observations[closing,arm*35+21:arm*35+27].reshape(-1,2,3)
            magnitude=torch.rand(len(f),2,1,device=f.device)*.20+(.30 if arm==0 else .36)
            observations[closing,arm*35+21:arm*35+27]=(f/f.norm(dim=-1,keepdim=True).clamp_min(1e-8)*magnitude).reshape(-1,6)
    return observations,targets

def fit_option_router(policy,paths,output,device,updates=10000,batch_size=1024,seed=42,jaw_inputs=False):
    """Fit causal learned transitions; preserve all movement and jaw weights."""
    torch.manual_seed(seed);torch.set_num_threads(8)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    net=load_policy(policy,device)
    if not isinstance(net,OptionPosePolicy) or net.options!=19 or not bool(net.latch_option_poses):
        raise ValueError('Router fitting requires the latched 19-option native full actor')
    for parameter in net.parameters():parameter.requires_grad_(False)
    arrays=[];offset=0
    for path in paths:
        with np.load(path,allow_pickle=False) as archive:
            metadata=json.loads(str(archive['metadata']))
            if metadata['schema']!=SCHEMA or metadata['skill']!='full' or metadata.get('assisted_grasp') is not False:
                raise ValueError('Native complete full demonstrations required')
            if metadata.get('collection_complete') is not True:raise ValueError('Collection must finish before router training')
            item={k:archive[k].copy() for k in ('obs','actions','stage','validation','episode')}
            if item['obs'].shape[1]!=OBS_DIM or not np.isfinite(item['obs']).all():raise ValueError('Invalid physical observations')
            if set(np.unique(item['episode']).tolist())!=set(metadata['episodes']):raise ValueError('Episode provenance mismatch')
            for ep in np.unique(item['episode']):
                selected=item['episode']==ep
                if len(np.unique(item['validation'][selected]))!=1:raise ValueError('Validation must hold out whole episodes')
            item['episode']+=offset;offset=int(item['episode'].max())+1;arrays.append(item)
    data={k:torch.as_tensor(np.concatenate([a[k] for a in arrays]),device=device) for k in arrays[0]}
    examples=causal_router_examples(data);indices=examples['indices'];context=examples['context'];label=examples['label']
    if bool(((label<0)|(label>=net.options)).any()):raise ValueError('Invalid full option labels')
    with torch.no_grad():
        goals=[]
        for start in range(0,len(indices),4096):
            part=slice(start,start+4096);_,heads=net.outputs(data['obs'][examples['entry'][part]])
            goals.append(heads[torch.arange(len(heads),device=device),context[part]])
        goal=torch.cat(goals)
    validation=data['validation'][indices].bool();train_ids=(~validation).nonzero().flatten();valid_ids=validation.nonzero().flatten()
    if not len(train_ids) or not len(valid_ids):raise ValueError('Whole validation episodes required')
    groups=[]
    for option in context[train_ids].unique():
        for changed in (False,True):
            group=train_ids[(context[train_ids]==option)&((label[train_ids]!=context[train_ids])==changed)]
            if len(group):groups.append(group)
    net.option_router=LearnedOptionRouter(net.options).to(device)
    net.option_router.router_uses_jaw_position.fill_(jaw_inputs)
    optimizer=torch.optim.Adam(net.option_router.parameters(),lr=1e-4);history=[]
    def logits(ids,calibrate=False):
        observations=data['obs'][indices[ids]];targets=goal[ids].clone()
        if calibrate:
            observations,targets=calibrate_router_context(observations,targets,context[ids],label[ids])
        return net.option_router(observations,context[ids],targets,net.mean,net.std,net.input_mask)
    for update in range(1,updates+1):
        ids=torch.cat([g[torch.randint(len(g),(max(1,batch_size//len(groups)),),device=device)] for g in groups])
        loss=nn.functional.cross_entropy(logits(ids),label[ids])
        loss+=nn.functional.cross_entropy(logits(ids,calibrate=True),label[ids])
        optimizer.zero_grad();loss.backward();optimizer.step()
        if update%500==0 or update==updates:
            with torch.no_grad():
                chosen=valid_ids[torch.randperm(len(valid_ids),device=device)[:8192]]
                scores=logits(chosen);accuracy=float((scores.argmax(-1)==label[chosen]).float().mean())
                change=label[chosen]!=context[chosen]
                change_accuracy=float((scores[change].argmax(-1)==label[chosen][change]).float().mean()) if bool(change.any()) else None
            history.append(dict(update=update,loss=float(loss.detach()),validation_accuracy=accuracy,validation_change_accuracy=change_accuracy))
            print(f'[AUTONOMY ROUTER BC] {update}/{updates} loss={float(loss.detach()):.6f} validation={accuracy:.4f} change={change_accuracy}',flush=True)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    state=torch.load(policy,map_location=device,weights_only=True);state['state_dict']=net.state_dict();torch.save(state,output/'policy.pt')
    config=json.loads((Path(policy).parent/'training.json').read_text())
    config.update(retrained=True,neural_parameters_unchanged=False,parent_policy=str(policy),physical_evaluation_required=True,
        implementation_change='Learned option router from measured state, previous learned option and its cached neural goal',
        option_router_updates=updates,option_router_learning_rate=1e-4,option_router_batch_size=batch_size,
        option_router_datasets=[str(path) for path in paths],option_router_history=history,
        option_router_labels='Causal observed transitions; no future progress labels or teacher at inference',
        option_router_absolute_position_inputs=False,option_router_goal_context_calibration=True,
        option_router_contact_features='Finger force norms/opposition, measured progress and own neural pose error; jaw position/reference optional; no absolute height or force direction',
        option_router_uses_jaw_position=jaw_inputs,
        option_router_calibration_states_are_physical_demonstrations=False,
        option_router_pressure_calibration='Offline counterfactual pressure on observed close transitions, preserving actual progress flags',
        movement_and_jaw_parameters_frozen=True,option_router_training_episodes=int(data['episode'][~data['validation'].bool()].unique().numel()),
        option_router_validation_episodes=int(data['episode'][data['validation'].bool()].unique().numel()))
    (output/'training.json').write_text(json.dumps(config,indent=2)+'\n')
    return net.eval()

def fit_jaw_rate(policy,paths,output,device,updates=6000,batch_size=1024,seed=42,pressure_supervision=False,pressure_min=.30,pressure_max=.40):
    """Freeze movement/gate weights; imitate one-step pressure feedback on CUDA."""
    torch.manual_seed(seed);torch.set_num_threads(8)
    net=load_policy(policy,device)
    if not isinstance(net,OptionPosePolicy):raise ValueError('Jaw feedback fitting requires an option actor')
    for parameter in net.parameters():parameter.requires_grad_(False)
    arrays=[]
    skills=set()
    for path in paths:
        with np.load(path,allow_pickle=False) as archive:
            metadata=json.loads(str(archive['metadata']))
            if metadata['schema']!=SCHEMA or metadata['skill'] not in ('lift','full') or metadata.get('assisted_grasp') is not False:
                raise ValueError('Native lift or full dataset required')
            skills.add(metadata['skill'])
            if metadata['skill']=='full':
                params=metadata.get('parameters',{})
                if not (params.get('receiver_pressure_min')==.36 and params.get('receiver_pressure_max')==.42 and params.get('donor_release_rate')==.02):
                    raise ValueError('Full pressure labels require the verified 0.36/0.42 N, 0.02 rad/s teacher recipe')
            arrays.append({k:archive[k].copy() for k in ('obs','actions','stage','validation')})
    if len(skills)!=1:raise ValueError('Do not mix pickup and full-stage labels')
    full=skills=={'full'}
    if full and (not pressure_supervision or net.options!=19):raise ValueError('Full pressure BC requires categorical 19-option actor')
    net.jaw_rate_head=JawRateHead(net.options,pressure_supervision,full).to(device);net.use_jaw_rate_head.fill_(True)
    label=full_teacher_pressure_rates if full else teacher_pressure_rates
    data={k:torch.as_tensor(np.concatenate([item[k] for item in arrays]),device=device) for k in arrays[0]}
    reference=data['obs'][:,[34,69]]*.5
    target=((data['actions'][:,[6,13]]+1)*.25-reference)/(.15*.01)
    target=target.clamp(-1,1)
    if pressure_supervision:target=label(data['obs'],data['actions'],data['stage'],pressure_min,pressure_max)
    train_ids=(~data['validation'].bool()).nonzero().flatten();valid_ids=data['validation'].bool().nonzero().flatten()
    if not len(train_ids) or not len(valid_ids):raise ValueError('Whole held-out episodes required')
    groups=[train_ids[data['stage'][train_ids]==stage] for stage in data['stage'][train_ids].unique()]
    optimizer=torch.optim.Adam(net.jaw_rate_head.parameters(),lr=3e-4);history=[]
    pressure_ids=train_ids[data['stage'][train_ids]>=3]
    opening_ids=train_ids[data['stage'][train_ids]<(10 if full else 3)]
    def objective(obs,stage,rates):
        if not pressure_supervision:return (net.jaw_rate_head(obs,stage)-rates).square().mean()
        labels=(rates[...,None]-net.jaw_rate_head.rates).abs().argmin(-1)
        return nn.functional.cross_entropy(net.jaw_rate_head(obs,stage,logits=True).reshape(-1,len(net.jaw_rate_head.rates)),labels.reshape(-1))
    for update in range(1,updates+1):
        ids=torch.cat([g[torch.randint(len(g),(max(1,batch_size//len(groups)),),device=device)] for g in groups])
        loss=objective(data['obs'][ids],data['stage'][ids].long(),target[ids])
        if pressure_supervision and len(pressure_ids):
            # Counterfactual force calibration states, NOT physical rollouts.
            # Cover the expert's decisions between contacts and overload so
            # a learned pressure response cannot extrapolate to more closing.
            sample=pressure_ids[torch.randint(len(pressure_ids),(256,),device=device)]
            calibration=data['obs'][sample].clone()
            amplitude=torch.rand(256,2,device=device)*.65
            calibration[:,21:27]=0;calibration[:,21]=amplitude[:,0];calibration[:,24]=-amplitude[:,1]
            if full:
                amplitude2=torch.rand(256,2,device=device)*.65
                calibration[:,56:62]=0;calibration[:,56]=amplitude2[:,0];calibration[:,59]=-amplitude2[:,1]
            rates=label(calibration,data['actions'][sample],data['stage'][sample],pressure_min,pressure_max)
            loss+=objective(calibration,data['stage'][sample].long(),rates)
        if pressure_supervision and len(opening_ids):
            # Recover the demonstrated open-jaw reference after a failed
            # approach. These are offline calibration states, not rollouts.
            sample=opening_ids[torch.randint(len(opening_ids),(256,),device=device)]
            references=torch.rand(256,2,device=device)*(.5 if full else .20)
            calibration=calibrate_opening_observations(data['obs'][sample],data['stage'][sample],references,full)
            rates=label(calibration,data['actions'][sample],data['stage'][sample],pressure_min,pressure_max)
            loss+=objective(calibration,data['stage'][sample].long(),rates)
        optimizer.zero_grad();loss.backward();optimizer.step()
        if update%500==0 or update==updates:
            with torch.no_grad():validation=float(objective(data['obs'][valid_ids],data['stage'][valid_ids].long(),target[valid_ids]))
            history.append(dict(update=update,loss=float(loss.detach()),validation_loss=validation))
            print(f'[AUTONOMY JAW BC] {update}/{updates} loss={float(loss.detach()):.7f} validation={validation:.7f}',flush=True)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    state=torch.load(policy,map_location=device,weights_only=True);state['state_dict']=net.state_dict()
    torch.save(state,output/'policy.pt')
    config=json.loads((Path(policy).parent/'training.json').read_text())
    config.update(jaw_feedback_algorithm='Behavior Cloning of one-step jaw reference change from measured forces',
        retrained=True,neural_parameters_unchanged=False,
        implementation_change='New learned incremental jaw head; frozen movement and option parameters',
        jaw_feedback_updates=updates,jaw_feedback_learning_rate=3e-4,jaw_feedback_batch_size=batch_size,
        jaw_feedback_datasets=[str(path) for path in paths],parent_policy=str(policy),
        jaw_supervision='Offline native expert pressure decisions with counterfactual force calibration' if pressure_supervision else 'Recorded absolute target relative to executed reference',
        jaw_rate_categorical=pressure_supervision,calibration_states_are_physical_demonstrations=False,
        opening_reference_calibration=pressure_supervision,
        receiver_approach_opening_calibration=full and pressure_supervision,
        coupled_arm_pressure_features=full,full_transfer_rate_class=full,
        pressure_label_band_n=[pressure_min,pressure_max] if pressure_supervision else None,
        movement_and_option_parameters_frozen=True,physical_evaluation_required=True,jaw_feedback_history=history)
    (output/'training.json').write_text(json.dumps(config,indent=2)+'\n')
    return net.eval()

def train(paths,output,device='cuda:0',updates=6000,batch_size=1024,seed=42,architecture='mlp',future_steps=0,phase_endpoint=False,latch_option_poses=False,pose_entry_supervision=False,exclude_joint_shortcuts=False):
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
    pose_targets=future_poses(data,future_steps,phase_endpoint or architecture in ('mixture','linear_options')) if architecture in ('pose','mixture','linear_options') else None
    if future_steps and architecture not in ('pose','mixture','linear_options'):data['actions']=future_actions(data,future_steps)
    train_ids=(~data['validation'].bool()).nonzero().flatten();valid_ids=data['validation'].bool().nonzero().flatten()
    if not len(train_ids) or not len(valid_ids):raise ValueError('Separate whole validation episodes required')
    mean=data['obs'][train_ids].mean(0);std=data['obs'][train_ids].std(0).clamp_min(.05)
    if architecture in ('mixture','linear_options'):
        mask=(data['obs'][train_ids].std(0)>=1e-4).float()
        if all(meta['skill']=='lift' for meta in metas):mask[35:70]=0;mask[7:19]=0;mask[77:83]=0
        if exclude_joint_shortcuts:mask[7:19]=0;mask[42:54]=0;mask[77:83]=0
        options=int(data['stage'].max())+1
        stages=data['stage'].clone()
        for episode in data['episode'].unique():
            ids=(data['episode']==episode).nonzero().flatten()
            future=ids[(torch.arange(len(ids),device=device)+future_steps).clamp(max=len(ids)-1)]
            stages[ids]=data['stage'][future]
        net=OptionPosePolicy(mean,std,mask,options,architecture=='linear_options').to(device)
        allowed,initial,counts=demonstrated_transitions(paths,options,device)
        net.allowed_transitions.copy_(allowed);net.initial_option.fill_(initial)
        net.latch_option_poses.fill_(latch_option_poses)
        if pose_entry_supervision and not latch_option_poses:
            raise ValueError('Entry-state pose supervision requires latched option poses')
        entry_obs=option_entry_observations(data,stages) if pose_entry_supervision else None
        pose_valid=completed_option_pose_mask(data,stages) if pose_entry_supervision else None
        with torch.no_grad():
            if net.linear_goals:net.pose_head.weight.zero_();net.jaw_head.weight.zero_()
            else:net.heads.weight.zero_()
            for stage in range(options):
                ids=train_ids[stages[train_ids]==stage]
                if len(ids):
                    average=pose_targets[ids].mean(0)
                    if net.linear_goals:
                        net.pose_head.bias[stage*14:stage*14+14].copy_(average.reshape(2,8)[:,:7].flatten())
                        net.jaw_head.bias[stage*2:stage*2+2].copy_(average.reshape(2,8)[:,7])
                    else:net.heads.bias[stage*16:stage*16+16].copy_(average)
        opt=torch.optim.Adam(net.parameters(),lr=1e-4)
        groups=[train_ids[stages[train_ids]==stage] for stage in range(options)]
        groups=[g for g in groups if len(g)]
        weight=torch.tensor([100.,100.,100.,10.,10.,10.,10.,3.]*2,device=device)
        history=[]
        for update in range(1,updates+1):
            ids=torch.cat([g[torch.randint(len(g),(max(1,batch_size//len(groups)),),device=device)] for g in groups])
            logits,heads=net.outputs(data['obs'][ids])
            pose=heads[torch.arange(len(ids),device=device),stages[ids]]
            if entry_obs is not None:
                _,entry_heads=net.outputs(entry_obs[ids])
                entry_pose=entry_heads[torch.arange(len(ids),device=device),stages[ids]].reshape(-1,2,8)
                # The cached XYZ/quaternion comes from entry. Jaw labels and
                # the learned gate still use current physical observations.
                pose=torch.cat((entry_pose[...,:7],pose.reshape(-1,2,8)[...,7:8]),-1).reshape(-1,16)
            pose_loss=((pose-pose_targets[ids]).square()*weight).mean(-1)
            if pose_valid is not None:
                included=pose_valid[ids].float()
                pose_loss=(pose_loss*included).sum()/included.sum().clamp_min(1.)
            else:pose_loss=pose_loss.mean()
            loss=pose_loss+.5*nn.functional.cross_entropy(logits,stages[ids])
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            if update%500==0 or update==updates:
                with torch.no_grad():
                    logits,heads=net.outputs(data['obs'][valid_ids]);accuracy=float((logits.argmax(-1)==stages[valid_ids]).float().mean())
                history.append(dict(update=update,loss=float(loss.detach()),validation_option_accuracy=accuracy))
                print(f'[AUTONOMY OPTIONS BC] {update}/{updates} loss={float(loss.detach()):.7f} heldout_option_accuracy={accuracy:.3f}',flush=True)
        torch.save(dict(schema=SCHEMA,architecture=architecture,state_dict=net.state_dict()),output/'policy.pt')
        (output/'training.json').write_text(json.dumps(dict(algorithm='Hierarchical Behavior Cloning',options=options,
            learned_option_gate=True,teacher_stage_at_inference=False,datasets=[str(p) for p in paths],
            transition_constraints='Observed training-episode transitions; no validation data',transition_counts=counts.cpu().tolist(),
            latch_option_poses=latch_option_poses,
            pose_entry_supervision=pose_entry_supervision,
            aborted_pose_endpoints_excluded=pose_entry_supervision,
            exclude_joint_shortcuts=exclude_joint_shortcuts,
            linear_pose_heads=net.linear_goals,
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
