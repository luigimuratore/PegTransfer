"""SAC physical state, free motor actions and measured progress; no stage controller.

Quaternions are wxyz. Positions are relative to the environment origin.
The actor sees the final destination, physical history and actuator references,
never a teacher phase, a waypoint, an episode clock or the previous reward.
"""
import itertools
import numpy as np
import torch

DT = .01
OBS_DIM, ACTION_DIM = 110, 14
JOINTS = ('psm_yaw_joint','psm_pitch_end_joint','psm_main_insertion_joint',
          'psm_tool_roll_joint','psm_tool_pitch_joint','psm_tool_yaw_joint')
SCHEMA = 'peg-free-sac-v2-1'

def canonical(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.where(q[..., :1] < 0, -q, q)

def conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)

def multiply(a, b):
    w = a[..., :1]*b[..., :1] - (a[..., 1:]*b[..., 1:]).sum(-1,keepdim=True)
    v = a[..., :1]*b[..., 1:]+b[..., :1]*a[..., 1:]+torch.cross(a[..., 1:],b[..., 1:],dim=-1)
    return torch.cat((w,v),-1)

def apply(q,v):
    t=2*torch.cross(q[...,1:],v,dim=-1)
    return v+q[...,:1]*t+torch.cross(q[...,1:],t,dim=-1)

def rotvec(q):
    q=canonical(q);length=q[...,1:].norm(dim=-1,keepdim=True)
    return q[...,1:]*(2*torch.atan2(length,q[...,:1])/length.clamp_min(1e-8))

def quaternion(v):
    angle=v.norm(dim=-1,keepdim=True)
    return canonical(torch.cat((torch.cos(angle/2),v*torch.sinc(angle/(2*torch.pi))/2),-1))

def bounded(v,maximum):
    limit=torch.as_tensor(maximum,device=v.device,dtype=v.dtype)
    if limit.ndim==v.ndim-1:limit=limit.unsqueeze(-1)
    return v*(limit/v.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp(max=1)

class MotorReference:
    def __init__(self,state):
        self.p=state['tip'].clone();self.q=canonical(state['tool_q'].clone())
        self.jaw=torch.full_like(state['jaws'][...,0],.15)

    def reset(self,state,ids):
        self.p[ids]=state['tip'][ids];self.q[ids]=canonical(state['tool_q'][ids]);self.jaw[ids]=.15

    def advance(self,action,state):
        a=action.reshape(-1,2,7).clamp(-1,1)
        # Tracking caps depend only on actual contact and tracking error. They do
        # not supply a task direction or hold an inactive arm for the actor.
        contact=state['forces'].norm(dim=-1).amax(-1)>.002
        self.p+=bounded(a[...,:3],1.)*(.025*DT)
        self.p=state['tip']+bounded(self.p-state['tip'],torch.where(contact,.0005,.003))
        self.p[...,0].clamp_(-.15,.15);self.p[...,1].clamp_(-.12,.12);self.p[...,2].clamp_(.004,.18)
        self.q=canonical(multiply(quaternion(bounded(a[...,3:6],1.)*DT),self.q))
        lead=rotvec(multiply(self.q,conjugate(state['tool_q'])))
        lead=bounded(lead,torch.where(contact,.04,.15))
        self.q=canonical(multiply(quaternion(lead),state['tool_q']))
        self.jaw=(self.jaw+a[...,6]*(.15*DT)).clamp(0,.5)

class Progress:
    def __init__(self,n,device,profile,gamma=.9995):
        self.n=n;self.device=device;self.gamma=gamma
        self.flags=torch.zeros(n,4,dtype=torch.bool,device=device)
        self.contact_age=torch.zeros(n,2,device=device)
        self.lift_age=torch.zeros(n,device=device);self.place_age=torch.zeros(n,device=device)
        self.anchor=torch.zeros(n,2,3,device=device);self.anchor_q=torch.zeros(n,2,4,device=device)
        self.anchor_q[...,0]=1
        self.potential=torch.zeros(n,device=device)
        self.hole=torch.tensor([*profile['hole_center_m'],0.],device=device)
        lo,hi=np.asarray(profile['bounds_m'])
        self.corners=torch.tensor(list(itertools.product(*zip(lo,hi))),device=device,dtype=torch.float32)
        self.z=torch.linspace(float(lo[2]),float(hi[2]),9,device=device)
        from geometry import radii_at
        # Inscribed-circle clearance is conservative: every accepted post fits
        # inside the mesh-derived hole at all nine checked heights.
        self.radius=torch.tensor([float(np.min(radii_at(profile,float(z))[0])) for z in self.z.cpu()],device=device)
        sites=[c['tip_local_m'] for c in profile['candidates'] if 'tip_local_m' in c]
        self.sites=torch.tensor(sites or [[*profile['hole_center_m'],float(hi[2])]],device=device)

    def reset(self,ids,state):
        self.flags[ids]=False;self.contact_age[ids]=0;self.lift_age[ids]=0;self.place_age[ids]=0
        self.anchor[ids]=state['relative'][ids];self.anchor_q[ids]=state['relative_q'][ids]
        self.potential[ids]=self.phi(state)[ids]

    def contact(self,state):
        f=state['forces'];norm=f.norm(dim=-1)
        oppose=(f[:,:,0]*f[:,:,1]).sum(-1)<-.25*norm[:,:,0]*norm[:,:,1]
        touching=(norm.amin(-1)>=.10)&oppose&state['contact_valid'].all(-1)
        return touching

    def placement(self,s):
        q=s['peg_q'];center=s['peg']+apply(q,self.hole.expand(self.n,-1))
        error=(center[:,:2]-s['target'][:,:2]).norm(dim=-1)
        tilt=torch.acos(apply(q,torch.tensor([0.,0.,1.],device=self.device).expand(self.n,-1))[:,2].clamp(-1,1))
        inv=conjugate(q);axis=apply(inv,torch.tensor([0.,0.,1.],device=self.device).expand(self.n,-1))
        base=torch.cat((s['target'][:,:2]-s['peg'][:,:2],torch.zeros(self.n,1,device=self.device)),1)
        local=apply(inv,base)
        az=axis[:,2];safe=torch.where(az.abs()>.001,az,torch.ones_like(az))
        height=(self.z[None]-local[:,2:3])/safe[:,None]
        cross=local[:,None,:]+height[:,:,None]*axis[:,None,:]
        clearance=(self.radius[None]-(cross[:,:,:2]-self.hole[:2]).norm(dim=-1)-.002/safe.abs()[:,None]).amin(-1)
        free=s['forces'].norm(dim=-1).amax((1,2))<.01
        ok=(error<.0015)&(tilt<.12)&(clearance>.0001)&(az.abs()>.001)&((s['bottom']-.010).abs()<.002)&(s['speed']<.01)&free
        return ok,error,clearance

    def phi(self,s):
        q=s['peg_q'][:,None,:].expand(-1,len(self.sites),-1)
        sites=apply(q,self.sites[None].expand(self.n,-1,-1))+s['peg'][:,None,:]
        distance=(s['tip'][:,:,None,:]-sites[:,None,:,:]).norm(dim=-1).amin(-1)
        lift=((s['bottom']-.010)/.035).clamp(0,1)
        target=(s['peg'][:,:2]+apply(s['peg_q'],self.hole.expand(self.n,-1))[:,:2]-s['target'][:,:2]).norm(dim=-1)
        f=self.flags
        return (2*torch.exp(-distance[:,0]/.025)*(~f[:,0]) + 3*lift*f[:,0]*(~f[:,1])
                +2*torch.exp(-distance[:,1]/.025)*f[:,1]*(~f[:,2])
                +4*torch.exp(-target/.04)*f[:,3])

    def update(self,s,action,skill='full'):
        was=self.flags.clone();contact=self.contact(s)
        start=contact&(self.contact_age==0)
        self.anchor=torch.where(start[...,None],s['relative'],self.anchor)
        self.anchor_q=torch.where(start[...,None],s['relative_q'],self.anchor_q)
        slip=(s['relative']-self.anchor).norm(dim=-1)
        turn=rotvec(multiply(s['relative_q'],conjugate(self.anchor_q))).norm(dim=-1)
        contact=contact&(slip<.0015)&(turn<.15)
        self.contact_age=torch.where(contact,self.contact_age+DT,0.)
        stable=self.contact_age>=.4
        self.flags[:,0]|=stable[:,0]
        lifted=stable[:,0]&(s['bottom']>=.043)
        self.lift_age=torch.where(lifted,self.lift_age+DT,0.)
        self.flags[:,1]|=(self.lift_age>=2.)&self.flags[:,0]
        self.flags[:,2]|=stable[:,1]&self.flags[:,1]
        released=s['forces'][:,0].norm(dim=-1).amax(-1)<.01
        self.flags[:,3]|=self.flags[:,2]&stable[:,1]&released
        placed,hole_error,clearance=self.placement(s)
        self.place_age=torch.where(placed&self.flags[:,3],self.place_age+DT,0.)
        full=self.place_age>=1.
        peak=s['forces'].norm(dim=-1).amax((1,2))
        failed=(peak>1.)|((s['bottom']<.004)&self.flags[:,1])|(s['peg'][:,:2].norm(dim=-1)>.20)
        finite=torch.isfinite(s['peg']).all(-1)&torch.isfinite(s['tip']).all((1,2))
        failed|=~finite
        success={'lift':self.flags[:,1],'handover':self.flags[:,3],'full':full}[skill]&~failed
        after=self.phi(s);after=torch.where(failed|success,0.,after)
        rewards=(after*self.gamma-self.potential)+((self.flags&~was).float()*torch.tensor([4.,8.,4.,12.],device=self.device)).sum(-1)
        rewards+=20*full.float()-10*failed.float()-.001-.002*action.square().mean(-1)
        self.potential=after
        return rewards,success,failed,dict(hole_error=hole_error,clearance=clearance,peak_force=peak,full_success=full,slip=slip)

def observe(s,ref,progress):
    values=[];scale=torch.tensor([1.,1.,.1,1.,1.,1.],device=s['peg'].device)
    vel_scale=torch.tensor([1.5,1.5,.04,1.5,1.5,1.5],device=s['peg'].device)
    for a in (0,1):
        values.extend((s['tip'][:,a]/.1,canonical(s['tool_q'][:,a]),s['joints'][:,a]/scale,
            s['joint_vel'][:,a]/vel_scale,s['jaws'][:,a]/.5,s['forces'][:,a].flatten(1),
            s['relative'][:,a]/.03,canonical(s['relative_q'][:,a]),(ref.p[:,a]-s['tip'][:,a])/.003,
            rotvec(multiply(ref.q[:,a],conjugate(s['tool_q'][:,a])))/.15,ref.jaw[:,a,None]/.5))
    values.extend((s['peg']/.1,canonical(s['peg_q']),s['linvel']/.05,s['angvel']/2.,s['bottom'][:,None]/.05,
        s['target']/.1,s['source']/.1,progress.flags.float(),(progress.contact_age/.4).clamp(max=1),
        (progress.lift_age[:,None]/2.).clamp(max=1),progress.place_age[:,None].clamp(max=1)))
    result=torch.cat(values,dim=1)
    assert result.shape[1]==OBS_DIM,result.shape
    return result
