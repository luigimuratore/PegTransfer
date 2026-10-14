"""Explicit, bounded grasp assistance for the neural curriculum.

No teacher stages, waypoints or object pose writes. Acquire from measured
opposed contacts; transfer/release from measured grasp and opening. Every
nonzero strength is assistance, never evidence of a native grasp.
"""
import math
import torch
from rl_core import apply,canonical,conjugate,multiply,rotvec,bounded,quaternion

SUPPORT_PROFILES = {
    'gentle': dict(stiffness=100., damping=.8, force_limit_n=.15, torque_limit_nm=.002),
    'firm': dict(stiffness=300., damping=1.5, force_limit_n=.4, torque_limit_nm=.004),
}


class GraspSpring:
    def __init__(self,env,strength,profile='gentle',placement=False):
        if not math.isfinite(strength) or not 0<=strength<=1:raise ValueError('Support strength must be 0..1')
        if profile not in SUPPORT_PROFILES:raise ValueError('Unknown support profile')
        if placement and not strength:raise ValueError('Placement assistance requires nonzero support strength')
        self.env=env;self.object=env.scene['object'];self.strength=strength
        self.profile=profile;self.settings=SUPPORT_PROFILES[profile];self.placement_enabled=placement
        n=env.num_envs;dev=env.device
        self.holder=torch.zeros(n,dtype=torch.long,device=dev)
        self.relative=torch.zeros(n,3,device=dev);self.relative_q=torch.zeros(n,4,device=dev);self.relative_q[:,0]=1
        self.acquire_jaw=torch.zeros(n,device=dev);self.force=torch.zeros(n,1,3,device=dev);self.torque=torch.zeros_like(self.force)
        self.steps=torch.zeros(n,dtype=torch.long,device=dev);self.acquisitions=self.steps.clone();self.releases=self.steps.clone()
        self.force_peak=torch.zeros(n,device=dev);self.torque_peak=self.force_peak.clone();self.work=self.force_peak.clone()
        self.placement_active=torch.zeros(n,dtype=torch.bool,device=dev)
        self.placement_steps=self.steps.clone();self.placement_force_peak=self.force_peak.clone()
        self.placement_torque_peak=self.force_peak.clone()
        self.target_world=torch.zeros(n,3,device=dev);self.hole=torch.zeros(3,device=dev)
        if placement:self.hole[:2]=torch.tensor(env.profile['hole_center_m'],device=dev)
        self.original_write=self.object.write_data_to_sim
        def write():
            self.apply();self.original_write()
        self.object.write_data_to_sim=write

    def acquire(self,mask,arm,state,reference):
        if not self.strength:return
        inverse=conjugate(state['tool_q'][:,arm-1])
        self.relative[mask]=apply(inverse,state['peg']-state['tip'][:,arm-1])[mask]
        self.relative_q[mask]=multiply(inverse,state['peg_q'])[mask]
        self.acquire_jaw[mask]=reference.jaw[mask,arm-1]
        self.holder[mask]=arm;self.acquisitions[mask]+=1

    def observe(self,state,progress,reference):
        if not self.strength:return
        contact=progress.contact(state)&(progress.contact_age>=.4)
        donor=contact[:,0]&progress.flags[:,0]&(self.holder==0)&(state['bottom']>.009)
        self.acquire(donor,1,state,reference)
        donor_open=reference.jaw[:,0]>self.acquire_jaw+.02
        receiver=contact[:,1]&progress.flags[:,2]&((self.holder==0)|((self.holder==1)&donor_open))
        self.acquire(receiver,2,state,reference)
        rows=torch.arange(len(self.holder),device=self.holder.device);arm=(self.holder-1).clamp_min(0)
        opened=reference.jaw[rows,arm]>self.acquire_jaw+.015
        free=state['forces'][rows,arm].norm(dim=-1).amax(-1)<.01
        self.release((self.holder>0)&opened&free)
        if self.placement_enabled:
            # An explicit virtual insertion guide, only after the learned
            # policy releases both fingers near the destination. It cannot
            # carry the object to the destination or replace the handover.
            self.target_world[:]=state['target']+self.env.scene.env_origins
            center=state['peg']+apply(state['peg_q'],self.hole.expand(len(rows),-1))
            error=(center[:,:2]-state['target'][:,:2]).norm(dim=-1)
            zaxis=apply(state['peg_q'],torch.tensor([0.,0.,1.],device=rows.device).expand(len(rows),-1))
            free_all=state['forces'].norm(dim=-1).amax((1,2))<.01
            near=progress.flags[:,3]&(self.holder==0)&free_all&(error<.008)&(state['bottom']>.008)&(state['bottom']<.040)&(zaxis[:,2]>.95)
            settled=(error<.0005)&(state['bottom']-.01).abs().lt(.002)&(state['speed']<.01)&(zaxis[:,2]>math.cos(.1))
            self.placement_active.copy_(near&~settled)

    def release(self,mask):
        self.releases[mask]+=(self.holder[mask]>0).long();self.holder[mask]=0
        self.force[mask]=0;self.torque[mask]=0
        self.object.set_external_force_and_torque(self.force,self.torque,is_global=True)

    def reset(self,ids):
        mask=torch.zeros_like(self.holder,dtype=torch.bool);mask[ids]=True;self.release(mask)
        self.placement_active[ids]=False
        for field in (self.steps,self.acquisitions,self.releases,self.force_peak,self.torque_peak,self.work,
                      self.placement_steps,self.placement_force_peak,self.placement_torque_peak):field[ids]=0

    def apply(self):
        if not self.strength:return
        obj=self.object.data;n=len(self.holder);rows=torch.arange(n,device=self.holder.device)
        position=[];orientation=[]
        for arm in (1,2):
            frame=self.env.scene[f'ee_{arm}_frame'].data
            position.append(frame.target_pos_w[:,0]);orientation.append(frame.target_quat_w[:,0])
        position=torch.stack(position,1);orientation=canonical(torch.stack(orientation,1))
        arm=(self.holder-1).clamp_min(0);q=orientation[rows,arm]
        target=position[rows,arm]+apply(q,self.relative)
        target_q=multiply(q,self.relative_q)
        settings=self.settings
        force=bounded(settings['stiffness']*(target-obj.root_pos_w)-settings['damping']*obj.root_lin_vel_w,settings['force_limit_n'])*self.strength
        torque=bounded(.01*rotvec(multiply(target_q,conjugate(obj.root_quat_w)))-.0001*obj.root_ang_vel_w,settings['torque_limit_nm'])*self.strength
        active=self.holder>0
        self.force[:,0]=torch.where(active[:,None],force,0.)
        self.torque[:,0]=torch.where(active[:,None],torque,0.)
        if self.placement_enabled:
            center=obj.root_pos_w+apply(obj.root_quat_w,self.hole.expand(n,-1))
            correction=self.target_world-center;correction[:,2]=0
            guide=100*correction-.8*obj.root_lin_vel_w;guide[:,2]=0
            guide=bounded(guide,.15)*self.strength
            # Remove small roll/pitch while preserving measured yaw.
            axis=apply(obj.root_quat_w,torch.tensor([1.,0.,0.],device=rows.device).expand(n,-1))
            yaw=torch.zeros(n,3,device=rows.device);yaw[:,2]=torch.atan2(axis[:,1],axis[:,0])
            upright=quaternion(yaw)
            guide_torque=bounded(.01*rotvec(multiply(upright,conjugate(obj.root_quat_w)))-.0001*obj.root_ang_vel_w,.002)*self.strength
            self.force[:,0]+=torch.where(self.placement_active[:,None],guide,0.)
            self.torque[:,0]+=torch.where(self.placement_active[:,None],guide_torque,0.)
            self.placement_steps+=self.placement_active.long()
            self.placement_force_peak.copy_(torch.maximum(self.placement_force_peak,torch.where(self.placement_active,guide.norm(dim=-1),0.)))
            self.placement_torque_peak.copy_(torch.maximum(self.placement_torque_peak,torch.where(self.placement_active,guide_torque.norm(dim=-1),0.)))
        self.object.set_external_force_and_torque(self.force,self.torque,is_global=True)
        self.steps+=active.long();self.force_peak.copy_(torch.maximum(self.force_peak,self.force[:,0].norm(dim=-1)))
        self.torque_peak.copy_(torch.maximum(self.torque_peak,self.torque[:,0].norm(dim=-1)))
        dt=self.env.physics_dt
        power=(self.force[:,0]*obj.root_lin_vel_w).sum(-1)+(self.torque[:,0]*obj.root_ang_vel_w).sum(-1)
        self.work+=power.abs()*dt

    def evidence(self,index=None):
        def values(t):return t.cpu().tolist() if index is None else t[index].item()
        return dict(type='bounded_grasp_spring',strength=self.strength,assisted_grasp=self.strength>0,
            profile=self.profile,stiffness_n_m=self.settings['stiffness'],damping_ns_m=self.settings['damping'],
            holder=values(self.holder),applied_physics_steps=values(self.steps),acquisitions=values(self.acquisitions),
            releases=values(self.releases),force_peak_n=values(self.force_peak),torque_peak_nm=values(self.torque_peak),
            absolute_work_j=values(self.work),force_limit_n=self.settings['force_limit_n']*self.strength,
            torque_limit_nm=self.settings['torque_limit_nm']*self.strength,
            placement_guide_enabled=self.placement_enabled,placement_guide_active=values(self.placement_active),
            placement_guide_steps=values(self.placement_steps),placement_force_peak_n=values(self.placement_force_peak),
            placement_torque_peak_nm=values(self.placement_torque_peak),
            object_pose_writes=False,object_velocity_writes=False,teacher_stage_inputs=False)

    def close(self):
        self.placement_active[:]=False
        self.release(torch.ones_like(self.holder,dtype=torch.bool));self.object.write_data_to_sim=self.original_write
