"""Measured pressure preload and loss detection. No simulator/object commands."""
import math
import numpy as np

from geometry import valid_contact_pair
from motion import rotation


class GripController:
    def __init__(self, mode, open_angle, target_force=.35, max_force=1.,
                 friction=.9, mass_kg=.005, preload_seconds=.4):
        values=(open_angle,target_force,max_force,friction,mass_kg,preload_seconds)
        if not all(math.isfinite(v) and v>0 for v in values):
            raise ValueError('Finite positive grip parameters required')
        if target_force+.05>=max_force:
            raise ValueError('Force target plus deadband must be below the force limit')
        # This is only a screening estimate; inclined/wedge contacts and actual
        # tangential forces still require the measured lift/slip experiment.
        if 2*friction*(target_force-.05)<2.5*mass_kg*9.81:
            raise ValueError('Pressure/friction setting has insufficient nominal weight margin')
        self.mode=mode;self.open_angle=open_angle;self.angle=open_angle
        self.target_force=target_force;self.max_force=max_force
        self.friction=friction;self.mass_kg=mass_kg
        self.preload_seconds=preload_seconds
        self.stable_seconds=0.;self.anchor=None;self.reference=None
        self.empty_seconds=0.;self.weak_seconds=0.;self.lost_seconds=0.

    def update(self,row,phase,dt):
        if not math.isfinite(dt) or dt<=0:
            raise ValueError('Finite positive control timestep required')
        force=np.asarray(row['finger_normal_force_n'])
        vectors=np.asarray(row['finger_force_vectors_n'])
        relative=np.asarray(row['peg_in_tool_m'])
        opposing=np.dot(*vectors)<-.25*np.prod(force)
        contact=bool((force>.01).all() and opposing and valid_contact_pair(row,self.mode))
        strong=bool(contact and (force>=self.target_force-.05).all())
        reason=None
        if force.max()>self.max_force:
            reason='excessive_finger_force'
        else:
            self._pressure_step(force,phase,dt)

        slip=0.;rotation_slip=0.
        if phase=='close':
            if not strong:
                self.stable_seconds=0.;self.anchor=None
            elif self.anchor is None:
                self.anchor=relative.copy();self.stable_seconds=dt
            elif np.linalg.norm(relative-self.anchor)>.0005:
                self.anchor=relative.copy();self.stable_seconds=0.
            else:
                self.stable_seconds+=dt
            ready=self.stable_seconds>=self.preload_seconds-1e-9
            if ready and self.reference is None:
                self.reference=(relative.copy(),np.asarray(row['peg_quat_in_tool']).copy())
            self.empty_seconds=self.empty_seconds+dt if self.angle<=1e-6 and not strong else 0.
            if self.empty_seconds>=1.:
                reason=reason or 'jaw_closed_without_secure_contact'
            move=False
        else:
            if self.reference is None:
                reason=reason or 'missing_pressure_preload'
            else:
                slip=float(np.linalg.norm(relative-self.reference[0]))
                dot=abs(float(np.dot(row['peg_quat_in_tool'],self.reference[1])))
                rotation_slip=float(2*np.arccos(np.clip(dot,0.,1.)))
                if slip>=.0015 or rotation_slip>=.15:
                    reason=reason or 'grasp_slipped'
            self.lost_seconds=0. if contact else self.lost_seconds+dt
            self.weak_seconds=0. if strong else self.weak_seconds+dt
            self.stable_seconds=self.stable_seconds+dt if strong else 0.
            if self.lost_seconds>=.4:
                reason=reason or 'finger_contact_lost'
            elif self.weak_seconds>=1.:
                reason=reason or 'grip_pressure_not_recovered'
            ready=strong
            move=bool(strong and self.stable_seconds>=.1-1e-9 and reason is None)
        return dict(jaw_target_rad=self.angle,pressure_ready=bool(ready and reason is None),
                    motion_allowed=move,reason=reason,force_target_n=self.target_force,
                    preload_stable_s=self.stable_seconds,slip_mm=slip*1000,
                    rotation_slip_rad=rotation_slip,
                    nominal_friction_capacity_n=float(self.friction*force.sum()),
                    peg_weight_n=self.mass_kg*9.81)

    def _pressure_step(self,force,phase,dt):
        if force.min()<self.target_force-.05:
            # Build pressure gently once either finger touches; use the faster
            # jaw motion only while crossing the empty gap.
            close_rate=.02 if force.max()>.03 else .15
            self.angle=max(0.,self.angle-close_rate*dt)
        elif force.min()>self.target_force+.05:
            # A single strong finger must not cause both jaws to open while
            # the weaker contact still cannot hold the peg.
            self.angle=min(self.open_angle,self.angle+.05*dt)

class VertexGripController(GripController):
    """Balance first contact by moving the real tool, keeping jaws coupled.

    A bounded translation along the pad closing axis unloads the first finger
    while bringing the other into contact. Stop increasing pressure during
    that correction. Never move the object or compensate an ejected peg.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.close_origin=None;self.peg_origin=None;self.axis=None
        self.center_offset=0.;self.balance_seconds=0.;self.balancing=False

    def _pressure_step(self,force,phase,dt):
        if not (phase=='close' and self.balancing):
            super()._pressure_step(force,phase,dt)

    def update(self,row,phase,dt):
        if not math.isfinite(dt) or dt<=0:
            raise ValueError('Finite positive control timestep required')
        force=np.asarray(row['finger_normal_force_n'])
        displacement=0.;center=None;abort=None
        if phase=='close':
            if self.close_origin is None:
                self.close_origin=np.asarray(row['tool_tip_w_m']).copy()
                self.peg_origin=np.asarray(row['peg_root_w_m']).copy()
                self.axis=rotation(row['tool_quaternion_wxyz']).as_matrix()[:,0]
            displacement=float(np.linalg.norm(np.asarray(row['peg_root_w_m'])-self.peg_origin))
            imbalance=float((force[1]-force[0])/max(force.sum(),1e-9))
            self.balancing=bool(force.max()>.015 and abs(imbalance)>.25)
            if self.balancing:
                self.center_offset=float(np.clip(self.center_offset+.0005*imbalance*dt,-.001,.001))
                self.balance_seconds+=dt
            else:
                self.balance_seconds=0.
            center=(self.close_origin+self.axis*self.center_offset).tolist()
            if displacement>=.001:
                abort='vertex_ejected_during_close'
            elif self.balance_seconds>=3.:
                abort='vertex_contact_balance_not_recovered'
        else:
            self.balancing=False
        result=super().update(row,phase,dt)
        if phase=='close' and self.balancing:
            self.stable_seconds=0.;self.anchor=None;self.reference=None
            result.update(pressure_ready=False,preload_stable_s=0.)
        if abort:
            result.update(reason=result['reason'] or abort,pressure_ready=False,motion_allowed=False)
        result.update(centering_position_w_m=center,centering_offset_mm=self.center_offset*1000,
                      jaw_closure_paused=self.balancing,closure_peg_displacement_mm=displacement*1000)
        return result


def grip_controller(mode,*args,**kwargs):
    controller=VertexGripController if mode=='vertex' else GripController
    return controller(mode,*args,**kwargs)
