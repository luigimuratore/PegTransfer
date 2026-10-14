"""Native full-task expert for curriculum data; no fixture/object servo."""
import torch
from autonomy_teacher import PickupTeacher
from rl_core import DT,apply,multiply,conjugate,canonical,rotvec,quaternion,bounded
from geometry import contact_candidate

FULL_STAGES=('above','outside','reach','close','lift','hold','transfer','receiver_above','receiver_outside',
    'receiver_reach','receiver_close','donor_release','donor_retreat','receiver_hold','transport','insert',
    'receiver_release','receiver_retreat','placement_settle')

class FullTeacher(PickupTeacher):
    def __init__(self,env):
        super().__init__(env);n=env.num_envs;dev=env.device
        mode=getattr(env,'receiver_mode','wall')
        plan=contact_candidate(env.profile,mode,0,.5,2,
            wall_angle_deg=getattr(env,'receiver_wall_angle',None),
            wall_depth_mm=getattr(env,'receiver_wall_depth',1.),
            wall_tilt_deg=getattr(env,'receiver_wall_tilt',20.),
            wall_open_angle=getattr(env,'receiver_open_angle',None))
        self.receiver_mode=mode;self.receiver_open=float(plan['open_angle_rad'])
        self.receiver_pressure_min=getattr(env,'receiver_pressure_min',.45)
        self.receiver_pressure_max=getattr(env,'receiver_pressure_max',.55)
        self.donor_release_rate=getattr(env,'donor_release_rate',.02)
        self.release_bottom=getattr(env,'receiver_release_bottom',.010)
        self.jaw[:,1]=self.receiver_open
        self.receiver_local=torch.tensor(plan['tip_local_m'],device=dev,dtype=torch.float32)
        self.receiver_q=torch.tensor(plan['tool_quaternion_local_wxyz'],device=dev,dtype=torch.float32)
        self.receiver_dir=torch.tensor(plan['approach_direction_local'],device=dev,dtype=torch.float32)
        self.goal=self.p.clone();self.goalq=self.q.clone()
        self.relative=env.state['relative'].clone();self.relative_q=env.state['relative_q'].clone()
        self.hold=self.p.clone();self.holdq=self.q.clone()
        self.receiver_anchor=self.p[:,1].clone();self.center=torch.zeros(n,3,device=dev)
        self.receiver_age=torch.zeros(n,device=dev);self.hold_age=self.receiver_age.clone()
        self.release_start=self.jaw.clone()
    def reset(self,ids):
        super().reset(ids)
        if hasattr(self,'goal'):
            self.goal[ids]=self.p[ids];self.goalq[ids]=self.q[ids]
            self.relative[ids]=self.env.state['relative'][ids];self.relative_q[ids]=self.env.state['relative_q'][ids]
            self.hold[ids]=self.p[ids];self.holdq[ids]=self.q[ids]
            self.center[ids]=0;self.receiver_age[ids]=0;self.hold_age[ids]=0
            self.jaw[ids,1]=self.receiver_open
    def strong(self):
        s=self.env.state
        threshold=s['forces'].new_tensor([.30,self.receiver_pressure_min])
        return (s['forces'].norm(dim=-1).amin(-1)>=threshold)&self.env.progress.contact(s)
    def pressure(self,arm,mask,rate=.15):
        f=self.env.state['forces'][:,arm].norm(dim=-1)
        close=torch.where(f.amax(-1)>.03,.02,rate)
        minimum=.30 if arm==0 else self.receiver_pressure_min
        maximum=.40 if arm==0 else self.receiver_pressure_max
        delta=torch.where(f.amin(-1)<minimum,-close*DT,torch.where(f.amin(-1)>maximum,.05*DT,0.))
        opening=.15 if arm==0 else self.receiver_open
        self.jaw[:,arm]=torch.where(mask,(self.jaw[:,arm]+delta).clamp(0,opening),self.jaw[:,arm])
    def action(self):
        advanced=self.stage>=6
        oldp=self.p.clone();oldq=self.q.clone();oldjaw=self.jaw.clone()
        base=super().action()
        base[:,13]=4*self.receiver_open-1
        self.p=torch.where(advanced[:,None,None],oldp,self.p)
        self.q=torch.where(advanced[:,None,None],oldq,self.q)
        self.jaw=torch.where(advanced[:,None],oldjaw,self.jaw)
        s=self.env.state;n=self.env.num_envs;stage=self.stage;dev=self.env.device
        strong=self.strong();self.goal=self.hold.clone();self.goalq=self.holdq.clone()
        hole=torch.tensor([*self.env.profile['hole_center_m'],0.],device=dev)
        center=s['peg']+apply(s['peg_q'],hole.expand(n,-1))
        transfer=s['tip'][:,0].clone();transfer[:,:2]+=torch.tensor([.020,-.007],device=dev)-center[:,:2]
        transfer[:,2]=self.hold[:,0,2]
        self.goal[:,0]=torch.where((stage==6)[:,None],transfer,self.goal[:,0])
        # Donor follows small REAL in-plane peg displacement while the receiver
        # closes; no pose command is ever applied to the object. Vertical
        # support remains at the measured carried height, so it cannot chase a drop.
        follower=s['peg']-apply(s['tool_q'][:,0],self.relative[:,0])
        follower[:,2]=self.hold[:,0,2]+(follower[:,2]-self.hold[:,0,2]).clamp(-.001,.001)
        follower=self.hold[:,0]+bounded(follower-self.hold[:,0],.002)
        followq=canonical(multiply(s['peg_q'],conjugate(self.relative_q[:,0])))
        correction=bounded(rotvec(multiply(followq,conjugate(self.holdq[:,0]))),.12)
        followq=canonical(multiply(quaternion(correction),self.holdq[:,0]))
        receiver_contact=(stage>=9)&(stage<=11)
        self.goal[:,0]=torch.where(receiver_contact[:,None],follower,self.goal[:,0])
        self.goalq[:,0]=torch.where(receiver_contact[:,None],followq,self.goalq[:,0])
        grasp=s['peg']+apply(s['peg_q'],self.receiver_local.expand(n,-1))
        orient=canonical(multiply(s['peg_q'],self.receiver_q.expand(n,-1)))
        outside=grasp+.015*apply(s['peg_q'],self.receiver_dir.expand(n,-1))
        above=outside.clone();above[:,2]=torch.maximum(grasp[:,2]+.025,torch.full_like(grasp[:,2],.055))
        self.goal[:,1]=torch.where((stage==7)[:,None],above,torch.where((stage==8)[:,None],outside,
            torch.where((stage==9)[:,None],grasp,self.goal[:,1])))
        self.goalq[:,1]=torch.where(((stage>=7)&(stage<=9))[:,None],orient,self.goalq[:,1])
        f=s['forces'][:,1].norm(dim=-1)
        imbalance=(f[:,1]-f[:,0])/f.sum(-1).clamp_min(1e-8)
        # Continue slow closure through very weak first contact, so the
        # second pad can reach the wall. Center only a loaded asymmetric grip.
        balancing=(stage==10)&(f.amax(-1)>.10)&(imbalance.abs()>.2)
        axis=apply(s['tool_q'][:,1],torch.tensor([1.,0.,0.],device=dev).expand(n,-1))
        # Center from the actual pad reaction, not an assumed finger index
        # convention. Validate this change physically before using its data.
        reaction=(s['forces'][:,1].sum(1)*axis).sum(-1)/f.sum(-1).clamp_min(1e-8)
        shift=.0005*DT*reaction[:,None]*axis*getattr(self.env,'balance_direction',1.)
        self.center=bounded(self.center+torch.where(balancing[:,None],shift,0.),.001)
        # Keep the receiver aligned with the measured wall while contacts
        # redistribute the load. A frozen world pose can torque the peg until
        # one pad loses contact. Bound this correction; never chase a fall.
        tracked=self.receiver_anchor+bounded(grasp-self.receiver_anchor,.002)
        tracked[:,2]=self.receiver_anchor[:,2]+(tracked[:,2]-self.receiver_anchor[:,2]).clamp(-.001,.001)
        close_position=tracked+self.center
        receiving=(stage==10)|(stage==11)
        self.goal[:,1]=torch.where(receiving[:,None],close_position,self.goal[:,1])
        turn=bounded(rotvec(multiply(orient,conjugate(self.holdq[:,1]))),.12)
        tracked_q=canonical(multiply(quaternion(turn),self.holdq[:,1]))
        self.goalq[:,1]=torch.where(receiving[:,None],tracked_q,self.goalq[:,1])
        self.pressure(0,(stage>=6)&(stage<=10))
        aligned=((s['tip'][:,1]-close_position).norm(dim=-1)<.0003)&(
            rotvec(multiply(tracked_q,conjugate(s['tool_q'][:,1]))).norm(dim=-1)<.03)
        self.pressure(1,(stage==10)&~balancing&aligned,rate=.02)
        self.pressure(1,(stage>=11)&(stage<=15))
        for arm,release in ((0,(stage>=11)&(stage<=12)),(1,stage>=16)):
            self.jaw[:,arm]=torch.where(release,(self.jaw[:,arm]+.15*DT).clamp(max=.5),self.jaw[:,arm])
        # Transfer load gradually. If receiver pressure weakens, pause/reclose
        # the donor instead of removing its support in a single rapid release.
        release_delta=torch.where(strong[:,1],self.donor_release_rate*DT,-self.donor_release_rate*DT)
        controlled=(oldjaw[:,0]+release_delta).clamp(max=.5)
        controlled=torch.maximum(controlled,self.release_start[:,0])
        self.jaw[:,0]=torch.where(stage==11,controlled,self.jaw[:,0])
        for arm,retreat in ((0,stage==12),(1,stage==17)):
            direction=self.direction if arm==0 else self.receiver_dir
            retreat_goal=self.hold[:,arm]+.015*apply(s['peg_q'],direction.expand(n,-1))
            self.goal[:,arm]=torch.where(retreat[:,None],retreat_goal,self.goal[:,arm])
        desiredq=canonical(conjugate(self.relative_q[:,1]))
        root=self.env.target.clone();root[:,:2]-=hole[:2]
        root[:,2]=.045-self.env.profile['bounds_m'][0][2]
        root[:,2]=torch.where(stage>=15,torch.full_like(root[:,2],self.release_bottom-self.env.profile['bounds_m'][0][2]+.0001),root[:,2])
        destination=root-apply(desiredq,self.relative[:,1])
        moving_receiver=(stage==14)|(stage==15)
        self.goal[:,1]=torch.where(moving_receiver[:,None],destination,self.goal[:,1])
        self.goalq[:,1]=torch.where(moving_receiver[:,None],desiredq,self.goalq[:,1])
        # Pressure gates affect ONLY this data-collection expert. Inference
        # executes the learned actor through generic actuator speed/limit caps.
        for arm in (0,1):
            moving=((stage==6) if arm==0 else (stage==14)|(stage==15))
            allowed=(~moving)|strong[:,arm]
            contact=((stage>=9)&(stage<=16)) if arm==1 else stage<=11
            speed=torch.where(moving,.003,torch.where(contact,.005,.025))
            nxt=self.p[:,arm]+bounded(self.goal[:,arm]-self.p[:,arm],speed*DT)
            nxt=s['tip'][:,arm]+bounded(nxt-s['tip'][:,arm],torch.where(contact,.0005,.003))
            self.p[:,arm]=torch.where((advanced&allowed)[:,None],nxt,self.p[:,arm])
            increment=bounded(rotvec(multiply(self.goalq[:,arm],conjugate(self.q[:,arm]))),torch.where(contact,.2,1.)*DT)
            nq=canonical(multiply(quaternion(increment),self.q[:,arm]))
            lead=bounded(rotvec(multiply(nq,conjugate(s['tool_q'][:,arm]))),torch.where(contact,.04,.15))
            nq=canonical(multiply(quaternion(lead),s['tool_q'][:,arm]))
            self.q[:,arm]=torch.where((advanced&allowed)[:,None],nq,self.q[:,arm])
        action=torch.cat(((self.p-s['tip'])/.003,rotvec(multiply(self.q,conjugate(s['tool_q'])))/.15,
                          (4*self.jaw-1)[...,None]),-1).reshape(n,14).clamp(-1,1)
        return torch.where(advanced[:,None],action,base)
    def update(self):
        stage=self.stage.clone();advanced=stage>=6;old_dwell=self.dwell.clone();super().update()
        self.dwell=torch.where(advanced,old_dwell,self.dwell)
        s=self.env.state;strong=self.strong();dev=self.env.device
        promoted=(stage==5)&self.env.progress.flags[:,1]
        self.stage=torch.where(promoted,6,self.stage)
        self.relative=torch.where(promoted[:,None,None],s['relative'],self.relative)
        self.relative_q=torch.where(promoted[:,None,None],s['relative_q'],self.relative_q)
        self.hold=torch.where(promoted[:,None,None],s['tip'],self.hold)
        self.holdq=torch.where(promoted[:,None,None],s['tool_q'],self.holdq)
        if not bool(advanced.any()):return
        error=(s['tip']-self.goal).norm(dim=-1)
        angle=rotvec(multiply(self.goalq,conjugate(s['tool_q']))).norm(dim=-1)
        reached=(error<.0007)&(angle<.05)
        reached[:,1]=torch.where(stage==9,(error[:,1]<.0002)&(angle[:,1]<.02),reached[:,1])
        active=torch.where((stage==6)|(stage==11)|(stage==12),0,1)
        arrival=reached[torch.arange(len(stage),device=dev),active]
        self.dwell=torch.where(advanced,torch.where(arrival,self.dwell+DT,0.),self.dwell)
        self.receiver_age=torch.where(strong[:,1],self.receiver_age+DT,0.)
        self.hold_age=torch.where((stage==13)&strong[:,1],self.hold_age+DT,0.)
        transition=(torch.isin(stage,torch.tensor([6,7,8,9,12,14,15,17],device=dev))&(self.dwell>=.10))
        transition|=(stage==10)&(self.receiver_age>=.4)&self.env.progress.flags[:,2]
        alone_age=getattr(self.env.progress,'receiver_alone_age',torch.zeros_like(self.receiver_age))
        supported_now=(alone_age>=2.)&strong[:,1]
        transition|=(stage==11)&self.env.progress.flags[:,3]&supported_now&(s['jaws'][:,0].amin(-1)>=.25)
        transition|=(stage==13)&(self.hold_age>=2.)
        transition|=(stage==16)&(s['forces'][:,1].norm(dim=-1).amax(-1)<.01)&(self.jaw[:,1]>=self.release_start[:,1]+.02)
        transition&=advanced
        self.stage=torch.where(transition,stage+1,self.stage)
        # The measured receiver-only hold already happened during release.
        # Avoid a second redundant timed dwell with identical physical state.
        self.stage=torch.where(transition&(stage==12)&supported_now,14,self.stage)
        # Capture actual arrival poses, never synthesize a carried peg pose.
        self.hold=torch.where(transition[:,None,None],s['tip'],self.hold)
        self.holdq=torch.where(transition[:,None,None],s['tool_q'],self.holdq)
        self.dwell=torch.where(transition,0.,self.dwell)
        close=transition&(stage==9)
        self.receiver_age=torch.where(close,0.,self.receiver_age)
        self.receiver_anchor=torch.where(close[:,None],s['tip'][:,1],self.receiver_anchor)
        self.center=torch.where(close[:,None],0.,self.center)
        got_receiver=transition&(stage==10)
        self.release_start[:,0]=torch.where(got_receiver,self.jaw[:,0],self.release_start[:,0])
        self.relative[:,1]=torch.where(got_receiver[:,None],s['relative'][:,1],self.relative[:,1])
        self.relative_q[:,1]=torch.where(got_receiver[:,None],s['relative_q'][:,1],self.relative_q[:,1])
        release=transition&(stage==15)
        self.release_start=torch.where(release[:,None],self.jaw,self.release_start)
