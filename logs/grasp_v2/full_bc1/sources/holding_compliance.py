"""Bounded robot visual compliance; never commands the object."""
import numpy as np
from motion import rotation,bounded_vector,PoseTrajectory


class HoldingCompliance:
    def __init__(self,row,reference):
        self.origin=np.array(row['tool_tip_w_m'])
        self.original_rotation=rotation(row['tool_quaternion_wxyz'])
        self.relative_position,self.relative_quaternion=reference
        self.trajectory=PoseTrajectory(row['tool_tip_w_m'],row['tool_quaternion_wxyz'])

    def command(self,row,dt):
        desired_rotation=rotation(row['peg_quaternion_wxyz'])*rotation(self.relative_quaternion).inv()
        delta=(desired_rotation*self.original_rotation.inv()).as_rotvec()
        from scipy.spatial.transform import Rotation
        desired_rotation=Rotation.from_rotvec(bounded_vector(delta,.08))*self.original_rotation
        desired_position=np.asarray(row['peg_root_w_m'])-desired_rotation.apply(self.relative_position)
        desired_position[2]=self.origin[2]  # Retain vertical support; never follow a falling peg.
        desired_position=self.origin+bounded_vector(desired_position-self.origin,.0005)
        return self.trajectory.advance(row['tool_tip_w_m'],row['tool_quaternion_wxyz'],
            desired_position,desired_rotation.as_quat()[[3,0,1,2]],dt,.002,.4,contact=True)
