"""Gentle, balanced receiver closure while the donor supports the peg."""
import numpy as np
from grip import GripController
from motion import rotation


class WallReceiverGrip(GripController):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.origin=None;self.axis=None;self.offset=0.;self.balancing=False;self.balance_seconds=0.

    def _pressure_step(self,force,phase,dt):
        if phase=='close':
            if not self.balancing:
                if force.min()<self.target_force-.05:self.angle=max(0.,self.angle-.02*dt)
                elif force.min()>self.target_force+.05:self.angle=min(self.open_angle,self.angle+.05*dt)
        else:super()._pressure_step(force,phase,dt)

    def update(self,row,phase,dt):
        force=np.asarray(row['finger_normal_force_n'])
        center=None
        if phase=='close':
            if self.origin is None:
                self.origin=np.array(row['tool_tip_w_m']);self.axis=rotation(row['tool_quaternion_wxyz']).as_matrix()[:,0]
            imbalance=(force[1]-force[0])/max(force.sum(),1e-9)
            self.balancing=bool(force.max()>.002 and abs(imbalance)>.2)
            if self.balancing:
                self.offset=float(np.clip(self.offset+.0005*imbalance*dt,-.001,.001))
                self.balance_seconds+=dt
            else:self.balance_seconds=0.
            center=(self.origin+self.axis*self.offset).tolist()
        else:self.balancing=False
        state=super().update(row,phase,dt)
        if phase=='close' and self.balancing:
            self.stable_seconds=0.;self.anchor=None;self.reference=None
            state.update(pressure_ready=False,preload_stable_s=0.)
        if self.balance_seconds>=3.:
            state.update(reason=state['reason'] or 'receiver_balance_not_recovered',motion_allowed=False)
        state.update(centering_position_w_m=center,centering_offset_mm=self.offset*1000,jaw_closure_paused=self.balancing)
        return state
