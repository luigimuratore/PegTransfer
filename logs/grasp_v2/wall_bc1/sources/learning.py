"""Contact-wall BC data and inference. No Isaac imports or object commands."""
import hashlib
import json
from pathlib import Path

import numpy as np
from motion import rotation, bounded_vector

SCHEMA='peg-wall-motor-bc-1'
STAGES=('settle','above','outside','reach','close','lift','hold','lower','release','retreat')
OBS_DIM=37
ACTION_DIM=4


def signature(args,profile,plan):
    folder=Path(__file__).parent
    files=('geometry.py','runtime.py','motion.py','grip.py','learning.py','probe.py','peg_profile.json','robot_kinematics.json')
    return dict(schema=SCHEMA,mode=args.mode,arm=args.arm,candidate=plan,
        source='L5',peg_asset_sha256=profile['asset_sha256'],robot_asset_sha256=profile['robot']['asset_sha256'],
        code_sha256=hashlib.sha256(b''.join((folder/name).read_bytes() for name in files)).hexdigest(),
        parameters={key:getattr(args,key) for key in
            ('mass_g','friction','contact_stiffness','contact_damping','jaw_torque','jaw_stiffness',
             'grip_force','preload_seconds','max_force','free_speed','contact_speed','lift_speed','rotation_speed')},
        control_dt_s=.01)


def observation(row,stage,goal,goal_quaternion,reference_position,jaw_command):
    tool_rot=rotation(row['tool_quaternion_wxyz'])
    error_rot=(rotation(goal_quaternion)*tool_rot.inv()).as_rotvec()
    values=np.r_[(np.asarray(goal)-row['tool_tip_w_m'])/.05,error_rot,
        (np.asarray(reference_position)-row['tool_tip_w_m'])/.003,
        np.asarray(row['peg_in_tool_m'])/.02,row['peg_quat_in_tool'],
        np.asarray(row['finger_force_vectors_n']).reshape(-1),np.asarray(row['jaw_rad'])/.5,
        jaw_command/.5,row['peg_bottom_m']/.05,row['peg_speed_m_s']/.02,
        np.eye(len(STAGES))[STAGES.index(stage)]]
    if values.shape!=(OBS_DIM,) or not np.isfinite(values).all():
        raise ValueError('Invalid measured wall-policy observation')
    return values.astype(np.float32)


def teacher_action(before_reference,after_reference,previous_jaw,next_jaw,dt,speed,allow_motion=True):
    # Speeds describe increments of the persistent reference, not servo lag.
    translation=(np.asarray(after_reference)-before_reference)/(dt*speed)
    translation=bounded_vector(translation,1.) if allow_motion else np.zeros(3)
    return np.r_[translation,np.clip((next_jaw-previous_jaw)/(dt*.15),-1.,1.)].astype(np.float32)


def motor_command(action,reference,actual,previous_jaw,dt,speed,contact=False,allow_motion=True):
    action=np.asarray(action)
    if action.shape!=(ACTION_DIM,) or not np.isfinite(action).all():
        raise ValueError('Non-finite or wrong-size neural action')
    velocity=bounded_vector(np.clip(action[:3],-1.,1.),1.)*speed
    pos=np.asarray(reference)+velocity*dt
    pos=np.asarray(actual)+bounded_vector(pos-actual,.0005 if contact else .003)
    if not allow_motion:
        pos=np.asarray(reference).copy()
    jaw=float(np.clip(previous_jaw+np.clip(action[3],-1.,1.)*.15*dt,0.,.5))
    return pos,jaw


def save_demonstrations(path,samples,cases,experiment):
    path=Path(path)
    accepted={c['episode'] for c in cases if c['complete'] and c['physical_grasp_hold_success']}
    rows=[sample for sample in samples if sample[0] in accepted]
    if not rows:
        raise ValueError('No complete physically successful wall episode; demonstrations not saved')
    if experiment['mode']!='wall':
        raise ValueError('The first BC dataset supports wall only')
    meta=dict(schema=SCHEMA,experiment=experiment,episodes=sorted(accepted),transitions=len(rows),
              assisted_grasp=False,physical_grasp_verified=True,teacher='programmed contact controller')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as file:
        np.savez_compressed(file,observations=np.stack([r[2] for r in rows]),
            actions=np.stack([r[3] for r in rows]),episode=np.array([r[0] for r in rows]),
            stage=np.array([STAGES.index(r[1]) for r in rows]),metadata=json.dumps(meta))
    return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**meta)


def load_demonstrations(path):
    with np.load(path,allow_pickle=False) as file:
        data={k:file[k].copy() for k in ('observations','actions','episode','stage')}
        meta=json.loads(str(file['metadata']))
    count=len(data['observations'])
    if meta['schema']!=SCHEMA or meta['experiment']['mode']!='wall' or meta['assisted_grasp']:
        raise ValueError('Incompatible/assisted demonstration dataset')
    if not meta['physical_grasp_verified'] or count!=meta['transitions']:
        raise ValueError('Physical success/transitions not verified')
    for key,shape in (('observations',(count,OBS_DIM)),('actions',(count,ACTION_DIM)),
                      ('episode',(count,)),('stage',(count,))):
        if data[key].shape!=shape or not np.isfinite(data[key]).all():
            raise ValueError(f'Invalid {key} dataset')
    if (abs(data['actions'])>1.00001).any() or set(np.unique(data['episode']))!=set(meta['episodes']):
        raise ValueError('Invalid actions/episode IDs')
    for episode in meta['episodes']:
        if set(data['stage'][data['episode']==episode])!=set(range(len(STAGES))):
            raise ValueError('Incomplete grasp sequence in demonstration episode')
    return data,meta


class MotorPolicy:
    def __init__(self,path,experiment=None):
        self.path=Path(path)
        with np.load(path,allow_pickle=False) as file:
            self.metadata=json.loads(str(file['metadata']))
            if self.metadata['schema']!=SCHEMA:
                raise ValueError('Wrong motor policy schema')
            if experiment is not None and self.metadata['experiment']!=experiment:
                raise ValueError('Policy differs from current geometry/control/physics; keep original code and settings')
            self.mean=file['mean'].copy();self.std=file['std'].copy()
            self.layers=[(file[f'w{i}'].copy(),file[f'b{i}'].copy()) for i in range(3)]
        if self.mean.shape!=(OBS_DIM,) or self.std.shape!=(OBS_DIM,) or (self.std<=0).any():
            raise ValueError('Invalid policy normalization')
        previous=OBS_DIM
        for i,(weight,bias) in enumerate(self.layers):
            size=ACTION_DIM if i==2 else 128
            if weight.shape!=(size,previous) or bias.shape!=(size,):
                raise ValueError('Invalid policy layer dimensions')
            previous=size
        if not all(np.isfinite(a).all() for a in [self.mean,self.std,*[a for layer in self.layers for a in layer]]):
            raise ValueError('Non-finite policy parameters')

    def predict(self,obs):
        x=(np.asarray(obs,dtype=np.float32)-self.mean)/self.std
        if x.shape!=(OBS_DIM,) or not np.isfinite(x).all():
            raise ValueError('Invalid neural input')
        for i,(w,b) in enumerate(self.layers):
            x=w@x+b
            x=np.tanh(x) if i==2 else np.maximum(x,0.)
        return x
