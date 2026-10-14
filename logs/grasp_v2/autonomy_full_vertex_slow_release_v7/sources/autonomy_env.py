"""Randomize valid reset conditions; no object commands during an episode."""
import torch
from rl_env import FreePegVecEnv
from rl_core import apply,quaternion,observe,MotorReference,Progress,bounded,canonical,multiply,rotvec,conjugate,DT

SOURCES={'L5':(0.,0.),'L6':(0.,-.030)}
TARGETS={'R1':(.042,-.015),'R2':(.042,.015)}

class AutonomyProgress(Progress):
    """Require a sustained receiver-only hold before certifying handover."""
    def __init__(self,n,device,profile,gamma=.9995):
        super().__init__(n,device,profile,gamma)
        self.receiver_alone_age=torch.zeros(n,device=device)
    def reset(self,ids,state):
        super().reset(ids,state);self.receiver_alone_age[ids]=0
    def update(self,s,action,skill='full'):
        was=self.flags.clone();old_potential=self.potential.clone();old_place_age=self.place_age.clone()
        _,_,failed,details=super().update(s,action,skill)
        donor_free=s['forces'][:,0].norm(dim=-1).amax(-1)<.01
        supported=(self.contact_age[:,1]>=.4)&donor_free&(s['bottom']>=.0375)&self.flags[:,1]&~failed
        self.receiver_alone_age=torch.where(supported,self.receiver_alone_age+DT,0.)
        self.flags[:,3]=was[:,3]|(self.receiver_alone_age>=2.)
        placed,_,_=self.placement(s)
        self.place_age=torch.where(placed&self.flags[:,3],old_place_age+DT,0.)
        full=self.place_age>=1.
        success={'lift':self.flags[:,1],'handover':self.flags[:,3],'full':full}[skill]&~failed
        after=torch.where(failed|success,0.,self.phi(s))
        reward=after*self.gamma-old_potential+((self.flags&~was).float()*torch.tensor([4.,8.,4.,12.],device=self.device)).sum(-1)
        reward+=20*full.float()-10*failed.float()-.001-.002*action.square().mean(-1)
        self.potential=after;details['full_success']=full
        return reward,success,failed,details

class ContactMotorReference(MotorReference):
    """Generic contact speed caps, without a task direction or phase."""
    def advance(self,action,state):
        a=action.reshape(-1,2,7).clamp(-1,1)
        peak=state['forces'].norm(dim=-1).amax(-1)
        contact=peak>.002
        desired=state['tip']+bounded(a[...,:3],1.)*.003
        speed=torch.where(contact,.003,.025)
        self.p+=bounded(desired-self.p,speed*DT)
        self.p=state['tip']+bounded(self.p-state['tip'],torch.where(contact,.001,.003))
        self.p[...,0].clamp_(-.15,.15);self.p[...,1].clamp_(-.12,.12);self.p[...,2].clamp_(.004,.18)
        desired_q=canonical(multiply(quaternion(bounded(a[...,3:6],1.)*.15),state['tool_q']))
        increment=bounded(rotvec(multiply(desired_q,conjugate(self.q))),torch.where(contact,.2,1.)*DT)
        self.q=canonical(multiply(quaternion(increment),self.q))
        lead=bounded(rotvec(multiply(self.q,conjugate(state['tool_q']))),torch.where(contact,.04,.15))
        self.q=canonical(multiply(quaternion(lead),state['tool_q']))
        jaw_target=(a[...,6]+1)*.25
        rate=torch.where(contact&(jaw_target<self.jaw),.02,.15)*DT
        self.jaw=(self.jaw+(jaw_target-self.jaw).clamp(-rate,rate)).clamp(0,.5)

class AutonomyEnv(FreePegVecEnv):
    def __init__(self,env,profile,skill='lift',episode_steps=5000,seed=42,jitter_mm=0.,yaw_deg=0.,sources=('L5',),targets=('R2',)):
        self.reset_rng=torch.Generator(device=env.device).manual_seed(seed)
        self.jitter=jitter_mm/1000;self.yaw=yaw_deg*torch.pi/180
        self.source_names=list(sources);self.target_names=list(targets)
        self.source_values=torch.tensor([SOURCES[s] for s in sources],device=env.device)
        self.target_values=torch.tensor([TARGETS[s] for s in targets],device=env.device)
        self.source=torch.zeros(env.num_envs,3,device=env.device);self.source[:,2]=.010
        self.target=self.source.clone();self.target[:,:2]=self.target_values[0]
        self.reset_records=[None]*env.num_envs
        super().__init__(env,profile,skill,episode_steps)
        self.progress=AutonomyProgress(env.num_envs,env.device,profile)
        self.reference=ContactMotorReference(self.state)
        self.physical_reset=env.reset
        def reset(*args,**kwargs):
            result=self.physical_reset(*args,**kwargs)
            ids=kwargs.get('env_ids')
            if ids is None:ids=torch.arange(env.num_envs,device=env.device)
            self.randomize(ids)
            return result
        env.reset=reset

    def gather(self):
        s=super().gather();s['source']=self.source;s['target']=self.target;return s

    def randomize(self,ids):
        n=len(ids);dev=self.device
        source_idx=torch.randint(len(self.source_values),(n,),generator=self.reset_rng,device=dev)
        target_idx=torch.randint(len(self.target_values),(n,),generator=self.reset_rng,device=dev)
        self.source[ids,:2]=self.source_values[source_idx];self.target[ids,:2]=self.target_values[target_idx]
        jitter=(torch.rand((n,2),generator=self.reset_rng,device=dev)*2-1)*self.jitter
        yaw=(torch.rand(n,generator=self.reset_rng,device=dev)*2-1)*self.yaw
        vector=torch.zeros(n,3,device=dev);vector[:,2]=yaw;q=quaternion(vector)
        center=torch.tensor([*self.profile['hole_center_m'],0.],device=dev)
        pose=self.env.scene['object'].data.root_state_w[ids,:7].clone()
        pose[:,:3]=self.origins[ids]+self.source[ids]-apply(q,center.expand(n,-1))
        pose[:,:2]+=jitter;pose[:,2]+=-self.profile['bounds_m'][0][2]+.0004
        pose[:,3:]=q
        self.env.scene['object'].write_root_pose_to_sim(pose,env_ids=ids)
        self.env.scene['object'].write_root_velocity_to_sim(torch.zeros(n,6,device=dev),env_ids=ids)
        self.env.scene.update(0.)
        for j,i in enumerate(ids.cpu().tolist()):
            self.reset_records[i]=dict(source=self.source_names[int(source_idx[j])],target=self.target_names[int(target_idx[j])],
                jitter_mm=(jitter[j]*1000).cpu().tolist(),yaw_deg=float(yaw[j]*180/torch.pi))
