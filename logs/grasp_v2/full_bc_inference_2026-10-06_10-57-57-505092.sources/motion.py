"""Bounded pose references and measured stall detection; no simulator imports."""
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation


def rotation(quaternion_wxyz):
    return Rotation.from_quat(np.asarray(quaternion_wxyz)[[1, 2, 3, 0]])


def bounded_vector(vector, maximum):
    length = np.linalg.norm(vector)
    return vector * min(1., maximum / max(length, 1e-12))


class PoseTrajectory:
    """Advance a persistent reference instead of restarting tiny errors at each tick.

    Reference speed is bounded. Tracking lead is also bounded, so an obstructed or
    unreachable pose cannot accumulate arbitrarily large actuator commands.
    """
    def __init__(self, position, quaternion):
        self.position = np.asarray(position, dtype=float).copy()
        self.orientation = rotation(quaternion)
        self.paused = False

    def advance(self, actual_position, actual_quaternion, goal_position, goal_quaternion,
                dt, linear_speed, angular_speed, contact=False, allow_motion=True):
        if not all(np.isfinite(x) and x > 0 for x in (dt, linear_speed, angular_speed)):
            raise ValueError('Finite positive timestep and speeds required')
        actual = np.asarray(actual_position)
        actual_rot = rotation(actual_quaternion)
        if not allow_motion:
            if not self.paused:
                self.position=actual.copy();self.orientation=actual_rot
            self.paused=True
            return self.position.copy(),self.orientation.as_quat()[[3,0,1,2]]
        if self.paused:
            self.position=actual.copy();self.orientation=actual_rot
        self.paused=False
        linear_lead = .0005 if contact else .003
        angular_lead = .04 if contact else .15
        next_pos = self.position + bounded_vector(np.asarray(goal_position)-self.position, linear_speed*dt)
        self.position = actual + bounded_vector(next_pos-actual, linear_lead)
        delta = (rotation(goal_quaternion)*self.orientation.inv()).as_rotvec()
        next_rot = Rotation.from_rotvec(bounded_vector(delta, angular_speed*dt))*self.orientation
        lead = (next_rot*actual_rot.inv()).as_rotvec()
        self.orientation = Rotation.from_rotvec(bounded_vector(lead, angular_lead))*actual_rot
        return self.position.copy(), self.orientation.as_quat()[[3, 0, 1, 2]]


def pose_reached(position_error_mm, rotation_error_rad):
    # Keep the original arrival accuracy; an above waypoint is never a grasp.
    return position_error_mm < .7 and rotation_error_rad < .05


class MotionWatchdog:
    """Fail after five simulated seconds with no meaningful pose-error reduction."""
    def __init__(self, seconds=5.):
        self.seconds = seconds
        self.history = deque()

    def stalled(self, elapsed, position_error_mm, rotation_error_rad):
        self.history.append((elapsed, position_error_mm, rotation_error_rad))
        while len(self.history) > 1 and self.history[1][0] <= elapsed-self.seconds:
            self.history.popleft()
        old_time, old_pos, old_rot = self.history[0]
        if elapsed-old_time < self.seconds-1e-8 or pose_reached(position_error_mm, rotation_error_rad):
            return False
        return old_pos-position_error_mm < .2 and old_rot-rotation_error_rad < .005
