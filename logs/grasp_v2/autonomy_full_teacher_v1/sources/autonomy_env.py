"""Randomize valid reset conditions; no object commands during an episode."""
import torch
from rl_env import FreePegVecEnv
from rl_core import apply,quaternion,observe

SOURCES={'L5':(0.,0.),'L6':(0.,-.030)}
TARGETS={'R1':(.042,-.015),'R2':(.042,.015)}

class AutonomyEnv(FreePegVecEnv):
    def __init__(self,env,profile,skill='lift',episode_steps=5000,seed=42,jitter_mm=0.,yaw_deg=0.,sources=('L5',),targets=('R2',)):
        self.reset_rng=torch.Generator(device=env.device).manual_seed(seed)
        self.jitter=jitter_mm/1000;self.yaw=yaw_deg*torch.pi/180
        self.source_names=list(sources);self.target_names=list(targets)
        self.source_values=torch.tensor([SOURCES[s] for s in sources],device=env.device)
        self.target_values=torch.tensor([TARGETS[s] for s in targets],device=env.device)
        self.source=torch.zeros(env.num_envs,3,device=env.device);self.source[:,2]=.010
        self.target=self.source.clone();self.target[:,:2]=self.target_values[0]
        self.reset_records=[None]*env.num_envs
        super().__init__(env,profile,skill,episode_steps)
        self.physical_reset=env.reset
        def reset(*args,**kwargs):
            result=self.physical_reset(*args,**kwargs)
            ids=kwargs.get('env_ids')
            if ids is None:ids=torch.arange(env.num_envs,device=env.device)
            self.randomize(ids)
            return result
        env.reset=reset

    def gather(self):
        s=super().gather();s['source']=self.source;s['target']=self.target;return s

    def randomize(self,ids):
        n=len(ids);dev=self.device
        source_idx=torch.randint(len(self.source_values),(n,),generator=self.reset_rng,device=dev)
        target_idx=torch.randint(len(self.target_values),(n,),generator=self.reset_rng,device=dev)
        self.source[ids,:2]=self.source_values[source_idx];self.target[ids,:2]=self.target_values[target_idx]
        jitter=(torch.rand((n,2),generator=self.reset_rng,device=dev)*2-1)*self.jitter
        yaw=(torch.rand(n,generator=self.reset_rng,device=dev)*2-1)*self.yaw
        vector=torch.zeros(n,3,device=dev);vector[:,2]=yaw;q=quaternion(vector)
        center=torch.tensor([*self.profile['hole_center_m'],0.],device=dev)
        pose=self.env.scene['object'].data.root_state_w[ids,:7].clone()
        pose[:,:3]=self.origins[ids]+self.source[ids]-apply(q,center.expand(n,-1))
        pose[:,:2]+=jitter;pose[:,2]+=-self.profile['bounds_m'][0][2]+.0004
        pose[:,3:]=q
        self.env.scene['object'].write_root_pose_to_sim(pose,env_ids=ids)
        self.env.scene['object'].write_root_velocity_to_sim(torch.zeros(n,6,device=dev),env_ids=ids)
        self.env.scene.update(0.)
        for j,i in enumerate(ids.cpu().tolist()):
            self.reset_records[i]=dict(source=self.source_names[int(source_idx[j])],target=self.target_names[int(target_idx[j])],
                jitter_mm=(jitter[j]*1000).cpu().tolist(),yaw_deg=float(yaw[j]*180/torch.pi))
