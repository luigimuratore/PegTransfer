"""Explicit temporary jaw aperture support; never chooses a task or a goal."""
import torch


class HeldJawGuard:
    def __init__(self,n,device,margin=.001,recovery_rate=.05):
        self.margin=margin;self.recovery_rate=recovery_rate
        self.stable_jaw=torch.zeros(n,2,device=device)
        self.valid=torch.zeros(n,2,device=device,dtype=torch.bool)
        self.interventions=torch.zeros(n,2,device=device,dtype=torch.long)

    def reset(self,ids=None):
        if ids is None:self.valid.zero_();self.stable_jaw.zero_()
        else:self.valid[ids]=False;self.stable_jaw[ids]=0.

    def apply(self,action,state,reference,support,recovery_only=False):
        force=state['forces'];norm=force.norm(dim=-1)
        opposition=(force[:,:,0]*force[:,:,1]).sum(-1)/norm.prod(-1).clamp_min(1e-8)
        held=support.holder[:,None]==torch.tensor([1,2],device=force.device)
        strong=held&(norm.amin(-1)>=.30)&(norm.amax(-1)<.65)&(opposition<-.5)
        if 'contact_valid' in state:strong &= state['contact_valid'].all(-1)
        # Once forces weaken, keep the last physically stable aperture, rather
        # than recording each new unsuccessful closure as a new stable grasp.
        self.stable_jaw=torch.where(strong,reference.jaw,self.stable_jaw)
        self.valid |= strong
        requested=(action.reshape(-1,2,7)[...,6]+1)*.25
        floor=(self.stable_jaw-self.margin).clamp(0,.5)
        closing=requested<=reference.jaw+1e-6
        protect=held&self.valid&closing&(requested<floor)&(norm.amax(-1)<.65)
        if recovery_only:
            protect&=support.recovery_active[:,None]&(norm[:,1].amax(-1)<.002)[:,None]
        recovery=torch.minimum(floor,reference.jaw+self.recovery_rate*.01)
        guarded=torch.where(protect,torch.maximum(requested,recovery),requested)
        result=action.clone().reshape(-1,2,7)
        result[...,6]=torch.where(protect,4*guarded-1,result[...,6])
        self.interventions+=protect.long()
        return result.flatten(1)

    def context(self):
        return torch.cat((self.stable_jaw/.5,self.valid.float()),1)

    def evidence(self):
        return dict(enabled=True,type='temporary_held_jaw_aperture_guard',
            margin_rad=self.margin,recovery_rate_rad_s=self.recovery_rate,
            stable_min_opposed_force_n=.30,interventions=self.interventions.tolist(),
            neural_opening_unchanged=True,task_phase_input=False,success_flags_written=False,
            object_pose_writes=False,object_velocity_writes=False)
