"""Optional bounded virtual spring for the full-task curriculum.

Applies real forces and torques in PhysX, never writes object pose/velocity.
Anchor acquisition is the caller's responsibility: only after measured grasp.
This is assistance and must never certify an unassisted physical grasp.
"""
import math


class PegSupport:
    def __init__(self,env,strength):
        import torch
        if not math.isfinite(strength) or not 0.<=strength<=1.:
            raise ValueError('Support strength must be finite and between zero and one')
        self.env=env;self.object=env.scene['object'];self.strength=strength
        self.holder=0;self.relative_pos=None;self.relative_quat=None
        self.steps=0;self.force_peak=0.;self.torque_peak=0.;self.work_abs_j=0.
        self.force=torch.zeros((1,1,3),device=env.device)
        self.torque=torch.zeros_like(self.force)
        self.original_write=self.object.write_data_to_sim
        def write():
            self.apply()
            self.original_write()
        self.object.write_data_to_sim=write

    def acquire(self,arm):
        from isaaclab.utils.math import subtract_frame_transforms
        if not self.strength:return
        frame=self.env.scene[f'ee_{arm}_frame'].data
        self.relative_pos,self.relative_quat=subtract_frame_transforms(
            frame.target_pos_w[:,0],frame.target_quat_w[:,0],
            self.object.data.root_pos_w,self.object.data.root_quat_w)
        self.relative_pos=self.relative_pos.clone();self.relative_quat=self.relative_quat.clone()
        self.holder=arm

    def release(self):
        self.holder=0;self.force.zero_();self.torque.zero_()
        self.object.set_external_force_and_torque(self.force,self.torque,is_global=True)

    def apply(self):
        import torch
        from isaaclab.utils.math import combine_frame_transforms,quat_mul,quat_inv,axis_angle_from_quat
        if not self.holder or not self.strength:
            return
        frame=self.env.scene[f'ee_{self.holder}_frame'].data
        pos,quat=combine_frame_transforms(frame.target_pos_w[:,0],frame.target_quat_w[:,0],
                                         self.relative_pos,self.relative_quat)
        obj=self.object.data
        delta=pos-obj.root_pos_w
        angle=axis_angle_from_quat(quat_mul(quat,quat_inv(obj.root_quat_w)))
        force=100.*delta-.8*obj.root_lin_vel_w
        torque=.01*angle-.0001*obj.root_ang_vel_w
        force*=torch.clamp(.15/force.norm(dim=-1,keepdim=True).clamp_min(1e-12),max=1.)
        torque*=torch.clamp(.002/torque.norm(dim=-1,keepdim=True).clamp_min(1e-12),max=1.)
        self.force[:]=force[:,None]*self.strength;self.torque[:]=torque[:,None]*self.strength
        self.object.set_external_force_and_torque(self.force,self.torque,is_global=True)
        self.steps+=1;self.force_peak=max(self.force_peak,float(self.force.norm()))
        self.torque_peak=max(self.torque_peak,float(self.torque.norm()))
        power=(self.force[:,0]*obj.root_lin_vel_w).sum()+(self.torque[:,0]*obj.root_ang_vel_w).sum()
        self.work_abs_j+=abs(float(power))*self.env.physics_dt

    def evidence(self):
        return dict(type='bounded_virtual_spring',strength=self.strength,holder=self.holder,
                    applied_physics_steps=self.steps,force_peak_n=self.force_peak,
                    torque_peak_nm=self.torque_peak,absolute_work_j=self.work_abs_j,
                    force_limit_n=.15*self.strength,torque_limit_nm=.002*self.strength,
                    translation_stiffness_n_m=100.*self.strength,
                    rotation_stiffness_nm_rad=.01*self.strength,
                    object_pose_writes=False,object_velocity_writes=False)
