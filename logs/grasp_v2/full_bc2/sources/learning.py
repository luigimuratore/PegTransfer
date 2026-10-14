"""Contact-wall BC data and inference. No Isaac imports or object commands."""
import hashlib
import json
from pathlib import Path

import numpy as np
from motion import rotation, bounded_vector

LEGACY_SCHEMA='peg-wall-motor-bc-1'
POSITION_SCHEMA='peg-wall-motor-bc-2'
SCHEMA='peg-wall-motor-bc-3'
SIGNATURE_FILES=('geometry.py','runtime.py','motion.py','grip.py','learning.py','probe.py','peg_profile.json','robot_kinematics.json')
CURRENT_SIGNATURE_FILES=SIGNATURE_FILES+('torch_backend.py',)
PHYSICS_FILES=('geometry.py','runtime.py','motion.py','grip.py','peg_profile.json','robot_kinematics.json')
STAGES=('settle','above','outside','reach','close','lift','hold','lower','release','retreat')
OBS_DIM=37
ACTION_DIM=4


def signature(args,profile,plan):
    folder=Path(__file__).parent
    return dict(schema=SCHEMA,mode=args.mode,arm=args.arm,candidate=plan,
        source='L5',peg_asset_sha256=profile['asset_sha256'],robot_asset_sha256=profile['robot']['asset_sha256'],
        code_sha256=hashlib.sha256(b''.join((folder/name).read_bytes() for name in CURRENT_SIGNATURE_FILES)).hexdigest(),
        parameters={key:getattr(args,key) for key in
            ('mass_g','friction','contact_stiffness','contact_damping','jaw_torque','jaw_stiffness',
             'grip_force','preload_seconds','max_force','free_speed','contact_speed','lift_speed','rotation_speed')},
        control_dt_s=.01)


def stage_speeds(parameters=None):
    p=parameters or {}
    free=p.get('free_speed',.025);near=p.get('contact_speed',.005);lift=p.get('lift_speed',.003)
    return np.array([lift if name in ('lift','hold') else
        near if name in ('reach','close','lower','release') else free for name in STAGES],dtype=np.float32)


def goal_action(obs,latent,parameters=None):
    """A positive learned gain on goal-reference error cannot invent a stop
    away from the reference target. Jaw velocity is still learned directly.
    This is a structured goal servo, not a free Cartesian velocity actor.
    """
    obs=np.asarray(obs);latent=np.asarray(latent)
    error=obs[..., :3]*.05-obs[...,6:9]*.003
    gain=5.+12.5*(1.+np.clip(latent[..., :3],-1.,1.))
    velocity=gain*error
    speed=stage_speeds(parameters)[np.argmax(obs[...,27:37],axis=-1)]
    length=np.linalg.norm(velocity,axis=-1)
    translation=velocity/np.maximum(length,speed)[...,None]
    return np.concatenate([translation,latent[...,3:4]],axis=-1).astype(np.float32)


def pressure_action(obs,parameters=None,open_angle=.15):
    """Measured pressure feedback in jaw velocity units; no time/success inputs.

    The learned actor scales this command by a positive gain. It cannot turn
    weak contact into a false zero-velocity equilibrium as bc2 did.
    """
    p=parameters or {};obs=np.asarray(obs);stage=np.argmax(obs[...,27:37],axis=-1)
    force=np.linalg.norm(obs[...,16:22].reshape(*obs.shape[:-1],2,3),axis=-1)
    minimum=force.min(axis=-1);maximum=force.max(axis=-1)
    target=p.get('grip_force',.35);previous=obs[...,24]*.5
    pressure=np.where(minimum<target-.05,np.where(maximum>.03,-.02/.15,-1.),
                      np.where(minimum>target+.05,.05/.15,0.))
    opening=np.where(previous<open_angle,1.,0.)
    action=np.where(np.isin(stage,[4,5,6,7]),pressure,opening)
    # Limit available jaw travel at the current control timestep.
    return np.clip(action,-previous/(.01*.15),np.maximum(0.,open_angle-previous)/(.01*.15))


def feedback_action(obs,latent,parameters=None,open_angle=.15):
    action=goal_action(obs,latent,parameters)
    gain=.5+.5*(1.+np.clip(np.asarray(latent)[...,3],-1.,1.))
    base=pressure_action(obs,parameters,open_angle)
    previous=np.asarray(obs)[...,24]*.5
    jaw=np.clip(base*gain,-1.,1.)
    jaw=np.clip(jaw,-previous/(.01*.15),np.maximum(0.,open_angle-previous)/(.01*.15))
    return np.concatenate([action[...,:3],jaw[...,None]],axis=-1).astype(np.float32)


def dataset_experiment(meta,initialize_from=None):
    """Upgrade old observation/action data only with a verified physics snapshot.
    Old data has the same 37 measured features and 4 executed velocity labels.
    Only the learned actor parameterization and probe integration may change.
    """
    experiment=meta['experiment']
    if 'parameters' not in experiment:
        if meta['schema']!=SCHEMA:
            raise ValueError('Legacy dataset lacks its physics signature')
        return experiment
    from types import SimpleNamespace
    from geometry import load_profile,contact_candidate
    profile=load_profile();old_plan=experiment['candidate']
    plan=contact_candidate(profile,'wall',old_plan['vertex'],old_plan['height_fraction'],experiment['arm'])
    args=SimpleNamespace(mode='wall',arm=experiment['arm'],**experiment['parameters'])
    current=signature(args,profile,plan)
    for key in ('mode','arm','source','candidate','peg_asset_sha256','robot_asset_sha256','parameters','control_dt_s'):
        if current[key]!=experiment[key]:
            raise ValueError(f'Demonstration physics/geometry differs: {key}; collect new demonstrations')
    if current==experiment:
        return current
    if initialize_from is None:
        raise ValueError('Older data requires --initialize_from CHECKPOINT to verify its saved physical sources')
    prior=MotorPolicy(initialize_from)
    if (prior.metadata['experiment']!=experiment
            and prior.metadata.get('demonstration_experiment')!=experiment):
        raise ValueError('Initial checkpoint does not match the demonstration experiment')
    folder=Path(__file__).parent;snapshot=Path(initialize_from).parent/'sources'
    prior_experiment=prior.metadata['experiment']
    files=CURRENT_SIGNATURE_FILES if prior_experiment['schema']==SCHEMA else SIGNATURE_FILES
    original_hash=hashlib.sha256(b''.join((snapshot/name).read_bytes() for name in files)).hexdigest()
    if original_hash!=prior_experiment['code_sha256']:
        raise ValueError('Checkpoint source snapshot fails its original experiment hash')
    for name in PHYSICS_FILES:
        if (snapshot/name).read_bytes()!=(folder/name).read_bytes():
            raise ValueError(f'Physical source changed: {name}; old demonstrations cannot be reused')
    return current


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
    if meta['schema'] not in (SCHEMA,POSITION_SCHEMA,LEGACY_SCHEMA) or meta['experiment']['mode']!='wall' or meta['assisted_grasp']:
        raise ValueError('Incompatible/assisted demonstration dataset')
    if not meta['physical_grasp_verified'] or count!=meta['transitions']:
        raise ValueError('Physical success/transitions not verified')
    for key,shape in (('observations',(count,OBS_DIM)),('actions',(count,ACTION_DIM)),
                      ('episode',(count,)),('stage',(count,))):
        if data[key].shape!=shape or not np.isfinite(data[key]).all():
            raise ValueError(f'Invalid {key} dataset')
    if (abs(data['actions'])>1.00001).any() or set(np.unique(data['episode']))!=set(meta['episodes']):
        raise ValueError('Invalid actions/episode IDs')
    if not np.issubdtype(data['stage'].dtype,np.integer) or ((data['stage']<0)|(data['stage']>=len(STAGES))).any():
        raise ValueError('Invalid stage IDs')
    if not np.allclose(data['observations'][:,27:37],np.eye(len(STAGES))[data['stage']],atol=1e-6):
        raise ValueError('Observation phase differs from recorded stage')
    for episode in meta['episodes']:
        if set(data['stage'][data['episode']==episode])!=set(range(len(STAGES))):
            raise ValueError('Incomplete grasp sequence in demonstration episode')
    return data,meta


class MotorPolicy:
    def __init__(self,path,experiment=None,device=None):
        self.path=Path(path)
        with np.load(path,allow_pickle=False) as file:
            self.metadata=json.loads(str(file['metadata']))
            if self.metadata['schema'] not in (SCHEMA,POSITION_SCHEMA,LEGACY_SCHEMA):
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
        self.torch_policy=None
        if device is not None:
            from torch_backend import TorchMotorPolicy
            self.torch_policy=TorchMotorPolicy(self,device)

    def predict(self,obs):
        if self.torch_policy is not None:
            return self.torch_policy.predict(obs)
        x=(np.asarray(obs,dtype=np.float32)-self.mean)/self.std
        if x.shape!=(OBS_DIM,) or not np.isfinite(x).all():
            raise ValueError('Invalid neural input')
        for i,(w,b) in enumerate(self.layers):
            x=w@x+b
            x=np.tanh(x) if i==2 else np.maximum(x,0.)
        params=self.metadata['experiment'].get('parameters')
        if self.metadata['schema']==SCHEMA:
            return feedback_action(obs,x,params,self.metadata['experiment'].get('candidate',{}).get('open_angle_rad',.15))
        return goal_action(obs,x,params) if self.metadata['schema']==POSITION_SCHEMA else x
