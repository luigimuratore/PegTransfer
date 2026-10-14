"""Explicit temporary jaw load relief; no task phase or object commands."""
import torch

class GripperLoadGuard:
    def __init__(self,n,device,ceiling=(.85,.65)):
        self.ceiling=torch.tensor(ceiling,device=device)
        self.interventions=torch.zeros(n,2,dtype=torch.long,device=device)
    def apply(self,action,state,reference):
        forces=state['forces'].norm(dim=-1).amax(-1)
        overload=forces>self.ceiling
        requested=(action.reshape(-1,2,7)[...,6]+1)*.25
        relief=(reference.jaw+.05*.01).clamp(max=.5)
        guarded=torch.where(overload,torch.maximum(requested,relief),requested)
        result=action.clone().reshape(-1,2,7)
        result[...,6]=4*guarded-1
        self.interventions+=overload.long()
        return result.reshape(-1,14)
    def evidence(self):
        return dict(enabled=True,type='temporary_measured_jaw_load_relief',ceiling_n=self.ceiling.cpu().tolist(),
            opening_rate_rad_s=.05,interventions=self.interventions.cpu().tolist(),
            task_phase_input=False,teacher=False,object_pose_writes=False,object_velocity_writes=False,
            physical_failure_limit_n=1.)
