"""Physical wall/wall transfer geometry and unchanged-limit route screening."""
import numpy as np
from scipy.spatial.transform import Rotation
from geometry import contact_candidate,radii_at
from kinematics import KinematicModel
from motion import rotation

TRANSFER_XY=np.array([.020,-.007])
TARGET_XY=np.array([.042,.015])


def transfer_wall_plan(profile,arm,depth_mm=1.,opening=.15):
    plan=contact_candidate(profile,'wall',2 if arm==1 else 0,.5,arm)
    if depth_mm==1. and opening==.15:return plan
    theta=np.radians(plan['wall_angle_deg']);z=profile['bounds_m'][1][2]-depth_mm/1000
    inner,outer=radii_at(profile,z)
    mid=(np.interp(theta,profile['angles_rad'],inner)+np.interp(theta,profile['angles_rad'],outer))/2
    xy=np.array(profile['hole_center_m'])+mid*np.array([np.cos(theta),np.sin(theta)])
    plan.update(tip_local_m=[*xy,z],open_angle_rad=opening,grasp_depth_mm=depth_mm)
    return plan


def wall_pose(root,peg_quaternion,plan):
    r=rotation(peg_quaternion)
    position=np.asarray(root)+r.apply(plan['tip_local_m'])
    orient=r*rotation(plan['tool_quaternion_local_wxyz'])
    return position,orient.as_quat()[[3,0,1,2]]


def approach_points(position,peg_quaternion,plan):
    outside=np.asarray(position)+rotation(peg_quaternion).apply(.015*np.array(plan['approach_direction_local']))
    above=outside.copy();above[2]=max(position[2]+.025,.055)
    return above,outside


def placement_evidence(row,profile,target=TARGET_XY):
    center=np.asarray(row['peg_root_w_m'])+rotation(row['peg_quaternion_wxyz']).apply([*profile['hole_center_m'],0.])
    xy=float(np.linalg.norm(center[:2]-target))
    tilt=float(np.arccos(np.clip(rotation(row['peg_quaternion_wxyz']).as_matrix()[2,2],-1,1)))
    # Check the entire post axis inside the tilted hole, including both ends.
    r=rotation(row['peg_quaternion_wxyz']);root=np.asarray(row['peg_root_w_m']);clearances=[]
    lo,hi=np.array(profile['bounds_m'])[:,2]
    for z in np.linspace(lo,hi,9):
        inverse=r.as_matrix().T
        wz=root[2]+(z-np.dot(inverse[2,:2],np.asarray(target)-root[:2]))/inverse[2,2]
        local=r.inv().apply(np.r_[target,wz]-root)
        delta=local[:2]-profile['hole_center_m'];theta=float(np.arctan2(delta[1],delta[0])%(2*np.pi))
        inner,_=radii_at(profile,float(z));angles=np.r_[profile['angles_rad'],2*np.pi]
        radius=np.interp(theta,angles,np.r_[inner,inner[0]])
        clearances.append(float(radius-np.linalg.norm(delta)-.002/max(abs(inverse[2,2]),1e-9)))
    clearance=min(clearances)
    passed=bool(xy<.0015 and tilt<.12 and clearance>.0001 and abs(row['peg_bottom_m']-.010)<.002
                and row['peg_speed_m_s']<.01 and max(row['finger_normal_force_n'])<.01)
    return dict(placed=passed,hole_error_mm=xy*1000,tilt_rad=tilt,bottom_m=row['peg_bottom_m'],
                minimum_post_hole_clearance_mm=clearance*1000)


def screen_full_route(profile,samples=8,plans=None):
    model=KinematicModel()
    root=np.array([-profile['hole_center_m'][0],0.,.010-profile['bounds_m'][0][2]])
    transfer=root+[TRANSFER_XY[0],TRANSFER_XY[1],.035]
    destination=root+[TARGET_XY[0],TARGET_XY[1],0.]
    plans=plans or [transfer_wall_plan(profile,arm) for arm in (1,2)]
    results=[]
    for arm,plan in enumerate(plans,1):
        grasp,quat=wall_pose(root if arm==1 else transfer,[1.,0,0,0],plan)
        above,outside=approach_points(grasp,[1.,0,0,0],plan)
        if arm==1:
            transfer_tip,_=wall_pose(transfer,[1.,0,0,0],plan)
            goals=[('above',above),('outside',outside),('reach',grasp),('lift',grasp+[0,0,.035]),
                   ('transfer',transfer_tip),('retreat',transfer_tip+.015*np.array(plan['approach_direction_local']))]
        else:
            placed,_=wall_pose(destination,[1.,0,0,0],plan)
            goals=[('above',above),('outside',outside),('reach',grasp),('transport',placed+[0,0,.035]),
                   ('insert',placed),('retreat',placed+.015*np.array(plan['approach_direction_local']))]
        base=np.array([-.07 if arm==1 else .07,0,.15])
        joints=np.array([.01,.01,.07,.01,.01,.01]);pose=model.forward(joints)
        start=base+pose[:3,3];start_rot=Rotation.from_matrix(pose[:3,:3])
        end_rot=rotation(quat);checked=0
        for stage,goal in goals:
            delta=(end_rot*start_rot.inv()).as_rotvec()
            for fraction in np.linspace(0,1,samples):
                orient=Rotation.from_rotvec(fraction*delta)*start_rot
                result=model.solve(start+fraction*(goal-start)-base,orient,joints)
                joints=result['joints'];checked+=1
                if not result['feasible']:
                    return dict(reachable=False,arm=arm,stage=stage,position_error_mm=result['position_error_m']*1000,
                                rotation_error_rad=result['rotation_error_rad'],samples_checked=checked)
            start=goal;start_rot=end_rot
        results.append(dict(arm=arm,samples_checked=checked))
    return dict(reachable=True,routes=results,transfer_hole_xy_m=TRANSFER_XY.tolist(),target_hole_xy_m=TARGET_XY.tolist(),
                note='Nominal IK only; two-arm contact and handover require physical validation')
