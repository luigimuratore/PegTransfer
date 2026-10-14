"""Higher travel caps on measured clearance; unchanged slow interactions.

This only bounds neural motor commands. It does not choose a waypoint,
grasp, release, or task phase. Physical joint limits and gains are unchanged.
"""
import math
import torch
from autonomy_env import ClearanceMotorReference


class FastTransitMotorReference(ClearanceMotorReference):
    def __init__(self, state, profile, free_scale=4., transport_scale=8.):
        if not math.isfinite(free_scale) or not 1 <= free_scale <= 12:
            raise ValueError('Free travel scale must be within 1..12')
        if not math.isfinite(transport_scale) or not 1 <= transport_scale <= 32:
            raise ValueError('Stable transport scale must be within 1..32')
        # Initialize the unchanged measured-clearance selector and histories.
        super().__init__(state, profile, min(free_scale, 4.), min(transport_scale, 4.), True)
        self.free_scale = free_scale
        self.transport_scale = transport_scale

    def settings(self):
        result = super().settings()
        result.update(profile='fast_travel', neural_translation_command_limit_mm_s=300.,
                      free_rotation_rad_s=max(1.,min(3.,self.free_scale/4.)),
                      clear_free_rotation_rad_s=max(1.,min(3.,self.free_scale/4.)),
                      rotation_caps_unchanged=self.free_scale<=4,
                      grasp_handover_insertion_caps_unchanged=True)
        return result

    def limits(self, state, contact):
        speed, angular, position_lead, rotation_lead=super().limits(state,contact)
        if self.free_scale>4:
            # Accelerate unloaded rotation only after the same measured
            # clearance and acceleration ramp permit fast translation.
            blend=((speed-.025)/(.025*(self.free_scale-1))).clamp(0,1)
            fast_angular=1+(min(3.,self.free_scale/4.)-1)*blend
            angular=torch.where(contact,angular,fast_angular)
        return speed,angular,position_lead,rotation_lead
