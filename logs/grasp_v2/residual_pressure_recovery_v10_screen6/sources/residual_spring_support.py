"""A firmer donor spring; original receiver compliance and wrench bounds."""
import torch


def install(support,donor_stiffness,vec=None,recovery_only=False):
    if not 300<=donor_stiffness<=1000:raise ValueError('Donor spring stiffness out of range')
    original=support.apply
    evidence=support.evidence
    # Keep this per-instance; never modify the archived global profile dictionary.
    support.settings=dict(support.settings,stiffness=float(donor_stiffness))
    def apply():
        donor=support.holder==1
        if recovery_only:
            norm=vec.state['forces'].norm(dim=-1)
            blend=((.35-norm[:,0].amin(-1))/.10).clamp(0,1)
            blend*=donor&(norm[:,1].amax(-1)<.002)
            stiffness=300.+(float(donor_stiffness)-300.)*blend
        else:stiffness=torch.where(donor,float(donor_stiffness),300.)
        support.settings['stiffness']=stiffness[:,None]
        try:original()
        finally:support.settings['stiffness']=float(donor_stiffness)
    def describe(index=None):
        row=evidence(index)
        row.update(stiffness_by_holder_n_m={'psm1':float(donor_stiffness),'psm2':300.},
                   temporary_donor_support=True,force_and_torque_limits_unchanged=True,
                   pressure_recovery_only=recovery_only,
                   pressure_recovery_force_band_n=[.25,.35] if recovery_only else None,
                   pressure_recovery_disabled_on_receiver_contact=recovery_only)
        return row
    support.apply=apply
    support.evidence=describe
