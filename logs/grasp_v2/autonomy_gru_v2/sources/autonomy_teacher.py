"""CUDA wall-pickup expert for data collection ONLY, never actor inference."""
import torch
from rl_core import DT, apply, canonical, multiply, conjugate, rotvec, quaternion, bounded
from geometry import contact_candidate

STAGES=('above','outside','reach','close','lift','hold')

class PickupTeacher:
    def __init__(self,env):
        self.env=env;n=env.num_envs;dev=env.device
        plan=contact_candidate(env.profile,'wall',2,.5,1)
        self.local=torch.tensor(plan['tip_local_m'],device=dev,dtype=torch.float32)
        self.orientation=torch.tensor(plan['tool_quaternion_local_wxyz'],device=dev,dtype=torch.float32)
        self.direction=torch.tensor(plan['approach_direction_local'],device=dev,dtype=torch.float32)
        self.stage=torch.zeros(n,device=dev,dtype=torch.long)
        self.dwell=torch.zeros(n,device=dev);self.pressure_age=self.dwell.clone()
        self.p=env.state['tip'].clone();self.q=env.state['tool_q'].clone()
        self.jaw=torch.full((n,2),.15,device=dev)
        self.pickup=self.p[:,0].clone();self.pickup_q=self.q[:,0].clone()
        self.anchor=env.state['relative'][:,0].clone()
        self.paused=torch.zeros(n,device=dev,dtype=torch.bool)
        self.reset(torch.arange(n,device=dev))

    def reset(self,ids):
        s=self.env.state
        self.stage[ids]=0;self.dwell[ids]=0;self.pressure_age[ids]=0
        self.p[ids]=s['tip'][ids];self.q[ids]=s['tool_q'][ids]
        self.jaw[ids]=.15;self.pickup[ids]=s['tip'][ids,0]
        self.pickup_q[ids]=s['tool_q'][ids,0];self.anchor[ids]=s['relative'][ids,0]
        self.paused[ids]=False

    def action(self):
        s=self.env.state;n=self.env.num_envs;stage=self.stage
        grasp=s['peg']+apply(s['peg_q'],self.local.expand(n,-1))
        orient=canonical(multiply(s['peg_q'],self.orientation.expand(n,-1)))
        outside=grasp+.015*apply(s['peg_q'],self.direction.expand(n,-1))
        above=outside.clone();above[:,2]=torch.maximum(grasp[:,2]+.025,torch.full_like(grasp[:,2],.055))
        goal=torch.where((stage==0)[:,None],above,torch.where((stage==1)[:,None],outside,grasp))
        goal=torch.where((stage==3)[:,None],self.pickup,goal)
        goal=torch.where((stage>=4)[:,None],self.pickup+torch.tensor([0.,0.,.035],device=s['peg'].device),goal)
        goalq=torch.where((stage>=3)[:,None],self.pickup_q,orient)
        f=s['forces'][:,0].norm(dim=-1)
        opposing=(s['forces'][:,0,0]*s['forces'][:,0,1]).sum(-1)<-.25*f.prod(-1)
        strong=(f.amin(-1)>=.30)&opposing&s['contact_valid'][:,0].all(-1)
        closing=stage>=3
        close_rate=torch.where(f.amax(-1)>.03,.02,.15)
        delta=torch.where(f.amin(-1)<.30,-close_rate*DT,
                          torch.where(f.amin(-1)>.40,torch.full_like(close_rate,.05*DT),0.))
        self.jaw[:,0]=torch.where(closing,(self.jaw[:,0]+delta).clamp(0,.15),.15)
        allowed=(stage<3)|((stage>=4)&strong)
        just_paused=(~allowed)&(~self.paused)
        resumed=allowed&self.paused
        reanchor=just_paused|resumed
        self.p[:,0]=torch.where(reanchor[:,None],s['tip'][:,0],self.p[:,0])
        self.q[:,0]=torch.where(reanchor[:,None],s['tool_q'][:,0],self.q[:,0])
        self.paused=~allowed
        contact=stage>=2
        speed=torch.where(stage>=4,.003,torch.where(contact,.005,.025))
        pnext=self.p[:,0]+bounded(goal-self.p[:,0],speed*DT)
        pnext=s['tip'][:,0]+bounded(pnext-s['tip'][:,0],torch.where(contact,.0005,.003))
        self.p[:,0]=torch.where(allowed[:,None],pnext,self.p[:,0])
        angular=torch.where(contact,.2,1.)*DT
        increment=bounded(rotvec(multiply(goalq,conjugate(self.q[:,0]))),angular)
        qnext=canonical(multiply(quaternion(increment),self.q[:,0]))
        lead=bounded(rotvec(multiply(qnext,conjugate(s['tool_q'][:,0]))),torch.where(contact,.04,.15))
        qnext=canonical(multiply(quaternion(lead),s['tool_q'][:,0]))
        self.q[:,0]=torch.where(allowed[:,None],qnext,self.q[:,0])
        # Position/orientation of the inactive PSM are labels to learn, not a
        # controller attached to inference. Its jaw remains open at .15 rad.
        pos=(self.p-s['tip'])/.003
        rot=rotvec(multiply(self.q,conjugate(s['tool_q'])))/.15
        return torch.cat((pos,rot,(4*self.jaw-1)[...,None]),-1).reshape(n,14).clamp(-1,1)

    def update(self):
        s=self.env.state;n=self.env.num_envs;stage=self.stage.clone()
        grasp=s['peg']+apply(s['peg_q'],self.local.expand(n,-1))
        orient=canonical(multiply(s['peg_q'],self.orientation.expand(n,-1)))
        outside=grasp+.015*apply(s['peg_q'],self.direction.expand(n,-1))
        above=outside.clone();above[:,2]=torch.maximum(grasp[:,2]+.025,torch.full_like(grasp[:,2],.055))
        goal=torch.where((stage==0)[:,None],above,torch.where((stage==1)[:,None],outside,grasp))
        goal=torch.where((stage>=4)[:,None],self.pickup+torch.tensor([0.,0.,.035],device=s['peg'].device),goal)
        goalq=torch.where((stage>=3)[:,None],self.pickup_q,orient)
        reached=((s['tip'][:,0]-goal).norm(dim=-1)<.0007)&(rotvec(multiply(goalq,conjugate(s['tool_q'][:,0]))).norm(dim=-1)<.05)
        self.dwell=torch.where(reached,self.dwell+DT,0.)
        f=s['forces'][:,0].norm(dim=-1)
        strong=(f.amin(-1)>=.30)&self.env.progress.contact(s)[:,0]
        start=strong&(self.pressure_age==0)
        self.anchor=torch.where(start[:,None],s['relative'][:,0],self.anchor)
        steady=(s['relative'][:,0]-self.anchor).norm(dim=-1)<.0005
        self.pressure_age=torch.where(strong&steady,self.pressure_age+DT,0.)
        advance=((stage<3)&(self.dwell>=.10))|((stage==3)&(self.pressure_age>=.40))|((stage==4)&(self.dwell>=.10))
        closed_arrival=advance&(stage==2)
        self.pickup=torch.where(closed_arrival[:,None],s['tip'][:,0],self.pickup)
        self.pickup_q=torch.where(closed_arrival[:,None],s['tool_q'][:,0],self.pickup_q)
        self.stage=torch.where(advance,stage+1,stage)
        self.dwell=torch.where(advance,0.,self.dwell)

