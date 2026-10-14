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


class FixedPegSupport:
    """Temporary PhysX fixed joint, outside the robot articulation.

    Retains physical collision contacts; breaks at a bounded force/torque.
    Removing the joint restores free dynamics without a pose/velocity write.
    """
    def __init__(self,env,strength):
        if not math.isfinite(strength) or not 0.<=strength<=1.:
            raise ValueError('Support strength must be 0..1')
        self.env=env;self.strength=strength;self.holder=0;self.joint=None
        self.acquisitions=0;self.releases=0;self.path='/World/TemporaryPegSupport';self.anchor_evidence=None

    def acquire(self,arm):
        from pxr import UsdPhysics,Gf,Sdf
        from isaaclab.utils.math import subtract_frame_transforms
        if not self.strength:return
        self.release()
        frame=self.env.scene[f'ee_{arm}_frame'].data;obj=self.env.scene['object'].data
        robot=self.env.scene[f'robot_{arm}'];indices,_=robot.find_bodies('psm_tool_tip_link')
        if len(indices)!=1:raise RuntimeError('Fixed support requires an identified physical tool body')
        index=indices[0];body_pos=robot.data.body_pos_w[:,index];body_quat=robot.data.body_quat_w[:,index]
        body_path=robot.root_physx_view.link_paths[0][index]
        p,q=subtract_frame_transforms(body_pos,body_quat,
                                      obj.root_pos_w,obj.root_quat_w)
        self.anchor_evidence=dict(body_path=body_path,body_position=body_pos[0].cpu().tolist(),
            tool_position=frame.target_pos_w[0,0].cpu().tolist(),relative_position=p[0].cpu().tolist(),
            relative_quaternion=q[0].cpu().tolist())
        p=p[0].cpu().tolist();q=q[0].cpu().tolist()
        joint=UsdPhysics.FixedJoint.Define(self.env.sim.stage,self.path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body_path)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path('/World/envs/env_0/Object')])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*p));joint.CreateLocalRot0Attr(Gf.Quatf(q[0],Gf.Vec3f(*q[1:])))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0));joint.CreateLocalRot1Attr(Gf.Quatf(1))
        joint.CreateExcludeFromArticulationAttr(True);joint.CreateCollisionEnabledAttr(True)
        joint.CreateBreakForceAttr(5.*self.strength);joint.CreateBreakTorqueAttr(.05*self.strength)
        joint.CreateJointEnabledAttr(True)
        self.joint=joint;self.holder=arm;self.acquisitions+=1

    def release(self):
        if self.joint is not None:
            self.joint.GetJointEnabledAttr().Set(False)
            self.env.sim.stage.RemovePrim(self.path)
            self.joint=None;self.releases+=1
        self.holder=0

    def evidence(self):
        return dict(type='temporary_physx_fixed_joint',strength=self.strength,holder=self.holder,
                    break_force_n=5.*self.strength,break_torque_nm=.05*self.strength,
                    acquisitions=self.acquisitions,releases=self.releases,
                    joint_authored_enabled=bool(self.joint and self.joint.GetJointEnabledAttr().Get()),
                    object_pose_writes=False,object_velocity_writes=False,
                    collision_contacts_retained=True,anchor=self.anchor_evidence)
