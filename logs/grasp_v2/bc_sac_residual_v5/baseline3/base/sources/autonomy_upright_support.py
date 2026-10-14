"""Temporary bounded roll/pitch support while fingers actually grasp the peg."""
import torch
from rl_core import apply,quaternion,canonical,multiply,conjugate,rotvec,bounded

def upright_torque(q,angular_velocity,mask):
    axis=apply(q,q.new_tensor([1.,0.,0.]).expand(len(q),-1))
    yaw=q.new_zeros(len(q),3);yaw[:,2]=torch.atan2(axis[:,1],axis[:,0])
    target=quaternion(yaw)
    moment=bounded(.03*rotvec(multiply(target,conjugate(canonical(q))))-.0003*angular_velocity,.004)
    return torch.where(mask[:,None],moment,0.)

def install(vec):
    support=vec.support
    if support is None or support.strength<=0:raise ValueError('Explicit nonzero grasp assistance required')
    original=support.apply
    support.upright_steps=torch.zeros_like(support.steps)
    support.upright_peak=torch.zeros_like(support.force_peak)
    def assisted():
        original()
        held=(support.holder>0)|vec.progress.contact(vec.state).any(-1)
        data=support.object.data
        torque=upright_torque(data.root_quat_w,data.root_ang_vel_w,held)*support.strength
        support.torque[:,0]=bounded(support.torque[:,0]+torque,.004*support.strength)
        support.object.set_external_force_and_torque(support.force,support.torque,is_global=True)
        support.upright_steps+=held.long()
        support.upright_peak.copy_(torch.maximum(support.upright_peak,torque.norm(dim=-1)))
    support.apply=assisted
    return lambda:dict(enabled=True,type='temporary_bounded_roll_pitch_torque',force_n=0.,
        torque_limit_nm=.004*support.strength,combined_grasp_torque_limit_nm=.004*support.strength,
        preserves_measured_yaw=True,only_measured_contact_or_acquired_holder=True,
        applied_physics_steps=support.upright_steps.cpu().tolist(),torque_peak_nm=support.upright_peak.cpu().tolist(),
        object_pose_writes=False,object_velocity_writes=False,success_flags_written=False,
        donor_acquisition='Measured opposed contact stable0.4s at bottom>9mm; before attempted lift')
