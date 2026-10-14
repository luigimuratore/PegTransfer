"""A firmer donor spring; original receiver compliance and wrench bounds."""
import torch


def install(support,donor_stiffness):
    if not 300<=donor_stiffness<=1000:raise ValueError('Donor spring stiffness out of range')
    original=support.apply
    evidence=support.evidence
    # Keep this per-instance; never modify the archived global profile dictionary.
    support.settings=dict(support.settings,stiffness=float(donor_stiffness))
    def apply():
        support.settings['stiffness']=torch.where(support.holder==1,
            float(donor_stiffness),300.)[:,None]
        try:original()
        finally:support.settings['stiffness']=float(donor_stiffness)
    def describe(index=None):
        row=evidence(index)
        row.update(stiffness_by_holder_n_m={'psm1':float(donor_stiffness),'psm2':300.},
                   temporary_donor_support=True,force_and_torque_limits_unchanged=True)
        return row
    support.apply=apply
    support.evidence=describe
