"""Randomize valid reset conditions; no object commands during an episode."""
import torch
from rl_env import FreePegVecEnv
from rl_core import apply,quaternion,observe,MotorReference,Progress,bounded,canonical,multiply,rotvec,conjugate,DT

SOURCES={'L5':(0.,0.),'L6':(0.,-.030),'L4':(0.,.030),'L2':(-.042,0.),'L1':(-.042,.030),'L3':(-.042,-.030)}
TARGETS={'R1':(.042,-.015),'R2':(.042,.015),'R3':(.084,.035),'R4':(.084,-.035),'R5':(.126,-.015),'R6':(.126,.015)}

class AutonomyProgress(Progress):
    """Require a sustained receiver-only hold before certifying handover."""
    def __init__(self,n,device,profile,gamma=.9995):
        super().__init__(n,device,profile,gamma)
        self.receiver_alone_age=torch.zeros(n,device=device)
        self.grasp_support=None
    def placement(self,s):
        ok,error,clearance=super().placement(s)
        if self.grasp_support is not None:
            ok=ok&(self.grasp_support.holder==0)&~self.grasp_support.placement_active
        return ok,error,clearance
    def reset(self,ids,state):
        super().reset(ids,state);self.receiver_alone_age[ids]=0
    def update(self,s,action,skill='full'):
        was=self.flags.clone();old_potential=self.potential.clone();old_place_age=self.place_age.clone()
        _,_,failed,details=super().update(s,action,skill)
        # A peg dropped back onto the board before transfer is a failed
        # attempt, even when the board catches it above the global floor.
        dropped=self.flags[:,1]&~was[:,3]&(s['bottom']<.025)&~self.contact(s).any(-1)
        failed|=dropped
        details['dropped_before_handover']=dropped
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
    def __init__(self,state,free_scale=1.,contact_scale=1.):
        import math
        if any(not math.isfinite(v) or not 1<=v<=2 for v in (free_scale,contact_scale)):
            raise ValueError('Motion speed scales must be finite and within 1..2')
        super().__init__(state)
        self.free_scale=free_scale;self.contact_scale=contact_scale
    def settings(self):
        return dict(free_scale=self.free_scale,contact_scale=self.contact_scale,
            free_translation_mm_s=25*self.free_scale,contact_translation_mm_s=3*self.contact_scale,
            free_rotation_rad_s=self.free_scale,contact_rotation_rad_s=.2*self.contact_scale,
            contact_jaw_close_rad_s=.02,jaw_open_rad_s=.15,control_dt_s=DT,
            joint_limits_changed=False,actuator_gains_changed=False,
            reference_speeds_not_guaranteed_actual_joint_speeds=True)
    def limits(self,state,contact):
        return (torch.where(contact,.003*self.contact_scale,.025*self.free_scale),
                torch.where(contact,.2*self.contact_scale,self.free_scale),
                torch.where(contact,.001,.003),torch.where(contact,.04,.15))
    def advance(self,action,state):
        a=action.reshape(-1,2,7).clamp(-1,1)
        peak=state['forces'].norm(dim=-1).amax(-1)
        contact=peak>.002
        desired=state['tip']+bounded(a[...,:3],1.)*.003
        speed,angular,position_lead,rotation_lead=self.limits(state,contact)
        self.p+=bounded(desired-self.p,speed*DT)
        self.p=state['tip']+bounded(self.p-state['tip'],position_lead)
        self.p[...,0].clamp_(-.15,.15);self.p[...,1].clamp_(-.12,.12);self.p[...,2].clamp_(.004,.18)
        desired_q=canonical(multiply(quaternion(bounded(a[...,3:6],1.)*.15),state['tool_q']))
        increment=bounded(rotvec(multiply(desired_q,conjugate(self.q))),angular*DT)
        self.q=canonical(multiply(quaternion(increment),self.q))
        lead=bounded(rotvec(multiply(self.q,conjugate(state['tool_q']))),rotation_lead)
        self.q=canonical(multiply(quaternion(lead),state['tool_q']))
        jaw_target=(a[...,6]+1)*.25
        rate=torch.where(contact&(jaw_target<self.jaw),.02,.15)*DT
        self.jaw=(self.jaw+(jaw_target-self.jaw).clamp(-rate,rate)).clamp(0,.5)

class ClearanceMotorReference(ContactMotorReference):
    """Experimental measured-clearance speed caps; never supplies a task goal.

    A single stable grasp can transport faster once the object clears the
    posts and the other tool is away. Contact approach, shared contact and
    insertion retain slow caps. No option, teacher stage or reward is read.
    """
    def __init__(self,state,profile,free_scale=2.,transport_scale=3.,interaction_compatible=False):
        import math
        if any(not math.isfinite(v) or not 1<=v<=4 for v in (free_scale,transport_scale)):
            raise ValueError('Clearance speed scales must be finite and within 1..4')
        super().__init__(state)
        self.free_scale=free_scale;self.transport_scale=transport_scale
        self.interaction_compatible=interaction_compatible
        self.bounds=torch.tensor(profile['bounds_m'],device=self.p.device,dtype=self.p.dtype)
        self.grip_age=torch.zeros_like(self.jaw)
        self.anchor=state['relative'].clone();self.anchor_q=state['relative_q'].clone()
        self.speed_cap=torch.full_like(self.jaw,.025 if interaction_compatible else .003)
    def reset(self,state,ids):
        super().reset(state,ids)
        self.grip_age[ids]=0;self.anchor[ids]=state['relative'][ids]
        self.anchor_q[ids]=state['relative_q'][ids];self.speed_cap[ids]=.025 if self.interaction_compatible else .003
    def settings(self):
        result=super().settings()
        result.update(profile='clearance',transport_scale=self.transport_scale,
            transport_translation_mm_s=3*self.transport_scale,
            near_translation_mm_s=5.,near_rotation_rad_s=.2,
            slow_peg_clearance_mm=6.,fast_peg_clearance_mm=25.,
            slow_board_tip_height_mm=30.,fast_board_tip_height_mm=55.,
            transport_bottom_height_mm=[30.,45.],other_tool_clearance_mm=[12.,30.],
            stable_opposed_contact_seconds=.4,stable_contact_min_force_n=.1,
            transport_peak_force_ceiling_n=.7,transport_rotation_rad_s=.2,
            free_acceleration_m_s2=.1,transport_acceleration_m_s2=.03,
            selector='Measured peg bounding box, board height, opposed forces and grasp stability; no task phase')
        if self.interaction_compatible:
            result.update(profile='travel',free_rotation_rad_s=1.,near_translation_mm_s=25.,near_rotation_rad_s=1.,
                transport_bottom_height_mm=[43.,50.],
                rotation_caps_unchanged=True,approach_minimum_speed_unchanged=True)
        return result
    def limits(self,state,contact):
        local=apply(conjugate(state['peg_q'])[:,None].expand(-1,2,-1),state['tip']-state['peg'][:,None])
        outside=torch.maximum(self.bounds[0]-local,local-self.bounds[1]).clamp_min(0)
        clearance=outside.norm(dim=-1)
        peg_blend=((clearance-.006)/.019).clamp(0,1)
        board_blend=((state['tip'][...,2]-.030)/.025).clamp(0,1)
        free_blend=torch.minimum(peg_blend,board_blend)
        free_speed=.005+(.025*self.free_scale-.005)*free_blend
        free_angular=.2+(self.free_scale-.2)*free_blend
        if self.interaction_compatible:
            # V49 changed orientation tracking and slowed unloaded approach.
            # Preserve both learned interaction regimes in this separate trial.
            free_speed=.025*(1+(self.free_scale-1)*free_blend)
            free_angular=torch.ones_like(free_speed)
        f=state['forces'];norm=f.norm(dim=-1)
        opposed=(f[:,:,0]*f[:,:,1]).sum(-1)<-.25*norm.prod(-1)
        strong=(norm.amin(-1)>=.1)&(norm.amax(-1)<.7)&opposed&state['contact_valid'].all(-1)
        start=strong&(self.grip_age==0)
        self.anchor=torch.where(start[...,None],state['relative'],self.anchor)
        self.anchor_q=torch.where(start[...,None],state['relative_q'],self.anchor_q)
        slip=(state['relative']-self.anchor).norm(dim=-1)
        turn=rotvec(multiply(state['relative_q'],conjugate(self.anchor_q))).norm(dim=-1)
        stable=strong&(slip<.0015)&(turn<.15)
        self.grip_age=torch.where(stable,(self.grip_age+DT).clamp(max=1.),0.)
        other_clear=((clearance.flip(-1)-.012)/.018).clamp(0,1)
        bottom_clear=((state['bottom']-.030)/.015).clamp(0,1)[:,None]
        if self.interaction_compatible:
            bottom_clear=((state['bottom']-.043)/.007).clamp(0,1)[:,None]
        single=(norm.amax(-1).flip(-1)<.002)&(self.grip_age>=.4)
        carry_blend=torch.minimum(other_clear,bottom_clear)*single.float()
        carry_speed=.003+(.003*self.transport_scale-.003)*carry_blend
        desired=torch.where(contact,carry_speed,free_speed)
        acceleration=torch.where(contact,.03,.1)
        self.speed_cap=torch.minimum(desired,self.speed_cap+acceleration*DT)
        if self.interaction_compatible:
            # A transient contact must not slow the next unloaded approach
            # below the original25 mm/s. Ramp only the additional speed.
            self.speed_cap=torch.where(contact,self.speed_cap,self.speed_cap.clamp_min(.025))
        return (self.speed_cap,torch.where(contact,.2,free_angular),
                torch.where(contact,.001,.003),torch.where(contact,.04,.15))

class AutonomyEnv(FreePegVecEnv):
    def __init__(self,env,profile,skill='lift',episode_steps=5000,seed=42,jitter_mm=0.,yaw_deg=0.,sources=('L5',),targets=('R2',),free_scale=1.,contact_scale=1.,motion_profile='contact',transport_scale=1.):
        self.reset_rng=torch.Generator(device=env.device).manual_seed(seed)
        self.jitter=jitter_mm/1000;self.yaw=yaw_deg*torch.pi/180
        self.source_names=list(sources);self.target_names=list(targets)
        self.source_values=torch.tensor([SOURCES[s] for s in sources],device=env.device)
        self.target_values=torch.tensor([TARGETS[s] for s in targets],device=env.device)
        self.source=torch.zeros(env.num_envs,3,device=env.device);self.source[:,2]=.010
        self.target=self.source.clone();self.target[:,:2]=self.target_values[0]
        self.reset_records=[None]*env.num_envs
        self.support=None
        self.paired_posts=False;self.pair_cursor=0
        super().__init__(env,profile,skill,episode_steps)
        self.progress=AutonomyProgress(env.num_envs,env.device,profile)
        if motion_profile=='contact':self.reference=ContactMotorReference(self.state,free_scale,contact_scale)
        elif motion_profile=='clearance':self.reference=ClearanceMotorReference(self.state,profile,free_scale,transport_scale)
        elif motion_profile=='travel':self.reference=ClearanceMotorReference(self.state,profile,free_scale,transport_scale,True)
        elif motion_profile=='fast_travel':
            from autonomy_fast_transit import FastTransitMotorReference
            self.reference=FastTransitMotorReference(self.state,profile,free_scale,transport_scale)
        else:raise ValueError('Unknown motion profile')
        self.physical_reset=env.reset
        def reset(*args,**kwargs):
            ids=kwargs.get('env_ids')
            if ids is None:ids=torch.arange(env.num_envs,device=env.device)
            if self.support is not None:self.support.reset(ids)
            result=self.physical_reset(*args,**kwargs)
            ids=kwargs.get('env_ids')
            if ids is None:ids=torch.arange(env.num_envs,device=env.device)
            self.randomize(ids)
            return result
        env.reset=reset

    def step_wait(self):
        if self.support is not None:self.support.observe(self.state,self.progress,self.reference)
        return super().step_wait()

    def close(self):
        if self.support is not None:self.support.close()
        super().close()

    def gather(self):
        s=super().gather();s['source']=self.source;s['target']=self.target;return s

    def randomize(self,ids):
        n=len(ids);dev=self.device
        source_idx=torch.randint(len(self.source_values),(n,),generator=self.reset_rng,device=dev)
        target_idx=torch.randint(len(self.target_values),(n,),generator=self.reset_rng,device=dev)
        if self.paired_posts:
            from autonomy_three_posts import pair_indices
            source_idx,self.pair_cursor=pair_indices(self.pair_cursor,n,len(self.source_names),dev)
            target_idx=source_idx
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
                jitter_mm=(jitter[j]*1000).cpu().tolist(),yaw_deg=float(yaw[j]*180/torch.pi),
                source_post_center_m=self.source[i].cpu().tolist(),target_post_center_m=self.target[i].cpu().tolist(),
                initial_hole_center_m=(self.source[i]+torch.cat((jitter[j],jitter.new_zeros(1)))).cpu().tolist(),
                reset_only_randomization=True,board_and_posts_unchanged=True)
