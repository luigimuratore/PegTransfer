"""Accelerate learned opening increments on empty, clear fingers only."""
import torch
from autonomy_fast_transit import FastTransitMotorReference
from rl_core import apply, conjugate


class FastOpenMotorReference(FastTransitMotorReference):
    def advance(self,action,state):
        previous=self.jaw.clone()
        super().advance(action,state)
        local=apply(conjugate(state['peg_q'])[:,None].expand(-1,2,-1),state['tip']-state['peg'][:,None])
        outside=torch.maximum(self.bounds[0]-local,local-self.bounds[1]).clamp_min(0)
        empty=state['forces'].norm(dim=-1).amax(-1)<.002
        clear=outside.norm(dim=-1)>.006
        # The actor still chooses whether to open. Add at most another
        # 0.15 rad/s until its reference reaches0.28rad. Closing is identical.
        # Actual finger speed retains the original PhysX0.25rad/s limit.
        extra=torch.minimum((self.jaw-previous).clamp_min(0),(.28-self.jaw).clamp_min(0))
        self.jaw=self.jaw+torch.where(empty&clear,extra,torch.zeros_like(extra))

    def settings(self):
        r=super().settings()
        r.update(empty_jaw_open_multiplier=2.,empty_open_boost_until_reference_rad=.28,
                 clear_empty_jaw_open_reference_rad_s=.3,actual_finger_velocity_limit_rad_s=.25,
                 empty_open_minimum_peg_clearance_mm=6.,closing_commands_unchanged=True,
                 contact_and_near_peg_opening_commands_unchanged=True)
        return r
