"""CUDA training/inference and the same measured feedback as the NumPy audit."""
import numpy as np
import torch
from learning import SCHEMA,POSITION_SCHEMA,stage_speeds


def resolve_device(device='auto'):
    requested='cuda:0' if device=='auto' else device
    target=torch.device(requested)
    if target.type=='cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable. Use the PegTransfer environment with GPU access; --device cpu is explicit fallback for checks only')
        torch.cuda.get_device_properties(target)
        torch.backends.cuda.matmul.allow_tf32=True
    return target


def decode_action(obs,latent,speeds,force_target=.35,open_angle=.15,pressure_feedback=True):
    stage=obs[...,27:37].argmax(dim=-1)
    error=obs[...,:3]*.05-obs[...,6:9]*.003
    velocity=(5.+12.5*(1.+latent[...,:3].clamp(-1.,1.)))*error
    translation=velocity/torch.maximum(velocity.norm(dim=-1),speeds[stage])[...,None]
    if not pressure_feedback:
        return torch.cat([translation,latent[...,3:4]],dim=-1)
    forces=obs[...,16:22].reshape(*obs.shape[:-1],2,3).norm(dim=-1)
    minimum=forces.amin(dim=-1);maximum=forces.amax(dim=-1)
    previous=obs[...,24]*.5
    pressure=torch.where(minimum<force_target-.05,torch.where(maximum>.03,-.02/.15,-1.),
                         torch.where(minimum>force_target+.05,.05/.15,0.))
    contact=(stage>=4)&(stage<=7)
    base=torch.where(contact,pressure,(previous<open_angle).to(obs.dtype))
    gain=.5+.5*(1.+latent[...,3].clamp(-1.,1.))
    jaw=(base*gain).clamp(-1.,1.)
    jaw=torch.maximum(-previous/(.01*.15),torch.minimum(jaw,(open_angle-previous).clamp_min(0.)/(.01*.15)))
    return torch.cat([translation,jaw[...,None]],dim=-1)


class TorchMotorPolicy:
    def __init__(self,policy,device='auto'):
        self.device=resolve_device(device)
        self.mean=torch.as_tensor(policy.mean,device=self.device,dtype=torch.float32)
        self.std=torch.as_tensor(policy.std,device=self.device,dtype=torch.float32)
        self.layers=[(torch.as_tensor(w,device=self.device,dtype=torch.float32),
                      torch.as_tensor(b,device=self.device,dtype=torch.float32)) for w,b in policy.layers]
        self.schema=policy.metadata['schema'];experiment=policy.metadata['experiment']
        params=experiment.get('parameters') or {}
        self.speeds=torch.as_tensor(stage_speeds(params),device=self.device)
        self.force_target=params.get('grip_force',.35)
        self.open_angle=experiment.get('candidate',{}).get('open_angle_rad',.15)

    @torch.inference_mode()
    def predict_tensor(self,obs):
        obs=torch.as_tensor(obs,device=self.device,dtype=torch.float32)
        x=(obs-self.mean)/self.std
        for i,(w,b) in enumerate(self.layers):
            x=torch.nn.functional.linear(x,w,b)
            x=x.tanh() if i==2 else x.relu()
        if self.schema in (SCHEMA,POSITION_SCHEMA):
            return decode_action(obs,x,self.speeds,self.force_target,self.open_angle,self.schema==SCHEMA)
        return x

    def predict(self,obs):
        obs=np.asarray(obs,dtype=np.float32)
        if obs.shape!=tuple(self.mean.shape) or not np.isfinite(obs).all():
            raise ValueError('Invalid neural input')
        return self.predict_tensor(obs).cpu().numpy()
