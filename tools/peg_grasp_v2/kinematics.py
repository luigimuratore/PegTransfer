"""USD joint-frame kinematics and constrained route screening; never starts Isaac."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from motion import rotation

MODEL=Path(__file__).with_name('robot_kinematics.json')


def joint_frames_from_usd(stage,UsdPhysics):
    """Export the actual joint constraints along base-to-tip, including fixed joints."""
    by_child={str(UsdPhysics.Joint(p).GetBody1Rel().GetTargets()[0]):p
              for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}
    chain=[];body='/psm/psm_tool_tip_link'
    while body!='/psm/psm_base_link':
        joint=by_child[body];chain.insert(0,joint)
        body=str(UsdPhysics.Joint(joint).GetBody0Rel().GetTargets()[0])
    def pose(position,quaternion):
        transform=np.eye(4)
        transform[:3,:3]=Rotation.from_quat([*quaternion.GetImaginary(),quaternion.GetReal()]).as_matrix()
        transform[:3,3]=position
        return transform.tolist()
    rows=[]
    for joint in chain:
        attr=lambda name:joint.GetAttribute('physics:'+name).Get()
        kind=joint.GetTypeName()
        row=dict(name=joint.GetName(),kind=kind,axis=attr('axis'),
                 parent_pose=pose(attr('localPos0'),attr('localRot0')),
                 child_pose=pose(attr('localPos1'),attr('localRot1')))
        if 'Fixed' not in kind:
            limits=[attr('lowerLimit'),attr('upperLimit')]
            row['limits']=np.radians(limits).tolist() if 'Revolute' in kind else limits
        rows.append(row)
    return rows


class KinematicModel:
    def __init__(self,path=MODEL):
        self.data=json.loads(Path(path).read_text())
        self.chain=self.data['chain']
        self.moving=[j for j in self.chain if 'limits' in j]
        self.names=[j['name'] for j in self.moving]
        self.limits=np.asarray([j['limits'] for j in self.moving])
        self.parents=[np.asarray(j['parent_pose']) for j in self.chain]
        self.children=[np.linalg.inv(j['child_pose']) for j in self.chain]
        # Keep physical limits unchanged and leave 2 degrees / 1 mm of margin.
        margin=np.array([.001 if 'Prismatic' in j['kind'] else np.radians(2.) for j in self.moving])
        self.bounds=np.column_stack([self.limits[:,0]+margin,self.limits[:,1]-margin])

    def forward(self,joints):
        q=np.array([joints[name] for name in self.names]) if isinstance(joints,dict) else np.asarray(joints)
        transform=np.eye(4);index=0
        for joint,parent,child in zip(self.chain,self.parents,self.children):
            delta=np.eye(4)
            if 'limits' in joint:
                axis=np.eye(3)['XYZ'.index(joint['axis'])]
                if 'Revolute' in joint['kind']:
                    delta[:3,:3]=Rotation.from_rotvec(axis*q[index]).as_matrix()
                else:
                    delta[:3,3]=axis*q[index]
                index+=1
            transform=transform@parent@delta@child
        return transform

    def jacobian(self,joints):
        """Geometric tip Jacobian in the base frame, for offline controller checks."""
        q=np.asarray(joints);transform=np.eye(4);axes=[];pivots=[];index=0
        for joint,parent,child in zip(self.chain,self.parents,self.children):
            transform=transform@parent
            delta=np.eye(4)
            if 'limits' in joint:
                axis=np.eye(3)['XYZ'.index(joint['axis'])]
                axes.append(transform[:3,:3]@axis);pivots.append(transform[:3,3].copy())
                if 'Revolute' in joint['kind']:
                    delta[:3,:3]=Rotation.from_rotvec(axis*q[index]).as_matrix()
                else:
                    delta[:3,3]=axis*q[index]
                index+=1
            transform=transform@delta@child
        result=np.zeros((6,len(self.moving)))
        for i,joint in enumerate(self.moving):
            if 'Revolute' in joint['kind']:
                result[:3,i]=np.cross(axes[i],transform[:3,3]-pivots[i])
                result[3:,i]=axes[i]
            else:
                result[:3,i]=axes[i]
        return result

    def solve(self,position,orientation,previous):
        """Multi-start constrained IK. Accept only measured-pose accuracy, never limit clipping."""
        def residual(q):
            pose=self.forward(q)
            return np.r_[100*(pose[:3,3]-position),
                         (Rotation.from_matrix(pose[:3,:3])*orientation.inv()).as_rotvec()]
        seeds=[previous]
        for roll in (0.,np.pi/2,-np.pi/2,np.pi,-np.pi):
            seed=np.asarray(previous).copy();seed[3]=roll;seeds.append(seed)
        best=None
        for seed in seeds:
            fit=least_squares(residual,np.clip(seed,self.bounds[:,0]+1e-8,self.bounds[:,1]-1e-8),
                              bounds=self.bounds.T,max_nfev=100,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            errors=residual(fit.x)
            pe=float(np.linalg.norm(errors[:3])/100)
            re=float(np.linalg.norm(errors[3:]))
            result=dict(joints=fit.x,position_error_m=pe,rotation_error_rad=re,
                        feasible=pe<.0002 and re<.02)
            if result['feasible']:
                return result
            if best is None or np.linalg.norm(errors)<best[0]:
                best=(float(np.linalg.norm(errors)),result)
        return best[1]


def route_check(profile,plan,free_speed=.025,contact_speed=.005,rotation_speed=1.,samples=12):
    model=KinematicModel()
    if model.data['asset_sha256']!=profile['robot']['asset_sha256']:
        raise ValueError('Robot kinematics asset hash differs from the inspected PSM')
    base=np.array([-.07 if plan['arm']==1 else .07,0.,.15])
    root=np.array([-profile['hole_center_m'][0],0.,.010-profile['bounds_m'][0][2]])
    grasp=root+plan['tip_local_m'];orient=rotation(plan['tool_quaternion_local_wxyz'])
    outside=grasp+.015*np.array(plan['approach_direction_local'])
    above=outside.copy();above[2]=max(grasp[2]+.025,.055)
    lift=grasp+[0,0,.035]
    q=np.array([.01,.01,.07,.01,.01,.01])
    pose=model.forward(q)
    position=base+pose[:3,3];start_rot=Rotation.from_matrix(pose[:3,:3])
    segments=[('above',above,False),('outside',outside,False),('reach',grasp,True),
              ('lift',lift,True),('lower',grasp,True),('retreat',above,False)]
    rows=[];max_pos=0.;max_rot=0.
    for name,goal,contact in segments:
        displacement=goal-position
        delta=(orient*start_rot.inv()).as_rotvec()
        linear_time=np.linalg.norm(displacement)/(contact_speed if contact else free_speed)
        angular_time=np.linalg.norm(delta)/(min(rotation_speed,.2) if contact else rotation_speed)
        duration=max(linear_time,angular_time)
        for t in np.linspace(0,duration,samples):
            pf=min(1.,t/max(linear_time,1e-12));rf=min(1.,t/max(angular_time,1e-12))
            target=position+pf*displacement
            target_rot=Rotation.from_rotvec(rf*delta)*start_rot
            result=model.solve(target-base,target_rot,q)
            q=result['joints']
            max_pos=max(max_pos,result['position_error_m']*1000)
            max_rot=max(max_rot,result['rotation_error_rad'])
            if not result['feasible']:
                return dict(checked=True,reachable=False,failed_stage=name,
                            position_error_mm=result['position_error_m']*1000,
                            rotation_error_rad=result['rotation_error_rad'],
                            joints=dict(zip(model.names,q.tolist())),samples_checked=len(rows)+1)
            rows.append(dict(stage=name,time_s=float(t),position_w_m=target.tolist(),
                             quaternion_wxyz=target_rot.as_quat()[[3,0,1,2]].tolist(),
                             joints=dict(zip(model.names,q.tolist()))))
        position=goal;start_rot=orient
    return dict(checked=True,reachable=True,samples_checked=len(rows),
                maximum_position_error_mm=max_pos,maximum_rotation_error_rad=max_rot,
                angular_limit_margin_deg=2.,insertion_limit_margin_mm=1.,
                note='Nominal sampled route with USD joint limits; not a swept collision or dynamics validation',
                route_samples=rows)
