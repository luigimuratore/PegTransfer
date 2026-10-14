"""CUDA physics wrapper for a free two-arm SAC actor; no teacher or fixture."""
import numpy as np
import torch
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env import VecEnv
from rl_core import *

class FreePegVecEnv(VecEnv):
    def __init__(self,env,profile,skill='lift',episode_steps=8000,gamma=.995):
        self.env=env;self.profile=profile;self.skill=skill;self.episode_steps=episode_steps
        self.origins=env.scene.env_origins;self.device=env.device
        self.ids=[];self.fingers=[]
        for arm in (1,2):
            robot=env.scene[f'robot_{arm}']
            self.ids.append([robot.joint_names.index(name) for name in JOINTS])
            self.fingers.append([robot.joint_names.index(f'psm_tool_gripper{f}_joint') for f in (1,2)])
        self.progress=Progress(env.num_envs,env.device,profile,gamma)
        self.state=self.gather();self.reference=MotorReference(self.state)
        self.age=torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
        self.episode_return=torch.zeros(env.num_envs,device=env.device)
        self.episode_peak=torch.zeros(env.num_envs,device=env.device)
        self.episodes=[]
        super().__init__(env.num_envs,Box(-np.inf,np.inf,(OBS_DIM,),dtype=np.float32),Box(-1.,1.,(ACTION_DIM,),dtype=np.float32))

    def gather(self):
        obj=self.env.scene['object'].data;n=self.env.num_envs
        tip=[];toolq=[];joints=[];vel=[];jaws=[];forces=[];valid=[]
        for i,arm in enumerate((1,2)):
            frame=self.env.scene[f'ee_{arm}_frame'].data;robot=self.env.scene[f'robot_{arm}'].data
            tip.append(frame.target_pos_w[:,0]-self.origins);toolq.append(frame.target_quat_w[:,0])
            joints.append(robot.joint_pos[:,self.ids[i]]);vel.append(robot.joint_vel[:,self.ids[i]])
            jaws.append(robot.joint_pos[:,self.fingers[i]].abs())
            f=[];v=[]
            for finger in (1,2):
                data=self.env.scene[f'contact_{arm}_{finger}'].data
                f.append(data.force_matrix_w[:,0,0]);v.append(torch.isfinite(data.contact_pos_w[:,0,0]).all(-1))
            forces.append(torch.stack(f,1));valid.append(torch.stack(v,1))
        tip=torch.stack(tip,1);toolq=canonical(torch.stack(toolq,1))
        peg=obj.root_pos_w-self.origins;pegq=canonical(obj.root_quat_w)
        world=apply(pegq[:,None].expand(-1,8,-1),self.progress.corners[None].expand(n,-1,-1))+peg[:,None]
        relative=apply(conjugate(toolq),peg[:,None].expand(-1,2,-1)-tip)
        relativeq=multiply(conjugate(toolq),pegq[:,None].expand(-1,2,-1))
        return dict(tip=tip,tool_q=toolq,joints=torch.stack(joints,1),joint_vel=torch.stack(vel,1),jaws=torch.stack(jaws,1),
            forces=torch.stack(forces,1),contact_valid=torch.stack(valid,1),relative=relative,relative_q=relativeq,
            peg=peg,peg_q=pegq,linvel=obj.root_lin_vel_w,angvel=obj.root_ang_vel_w,
            bottom=world[:,:,2].amin(-1),speed=obj.root_lin_vel_w.norm(dim=-1),
            target=torch.tensor([.042,.015,.010],device=self.device).expand(n,-1),
            source=torch.tensor([0.,0.,.010],device=self.device).expand(n,-1))

    def reset(self):
        self.env.reset();self.state=self.gather();ids=torch.arange(self.num_envs,device=self.device)
        self.reference.reset(self.state,ids);self.progress.reset(ids,self.state)
        self.age.zero_();self.episode_return.zero_();self.episode_peak.zero_()
        return observe(self.state,self.reference,self.progress).cpu().numpy()

    def step_async(self,actions):self.pending=actions

    def step_wait(self):
        from isaaclab.utils.math import subtract_frame_transforms
        action=torch.as_tensor(self.pending,device=self.device,dtype=torch.float32)
        if not torch.isfinite(action).all():raise ValueError('Non-finite motor action')
        self.reference.advance(action,self.state)
        command=torch.empty((self.num_envs,16),device=self.device)
        for a in (0,1):
            robot=self.env.scene[f'robot_{a+1}'].data
            p,q=subtract_frame_transforms(robot.root_pos_w,robot.root_quat_w,
                self.reference.p[:,a]+self.origins,self.reference.q[:,a])
            command[:,a*8:a*8+7]=torch.cat((p,q),1)
            command[:,a*8+7]=4*self.reference.jaw[:,a]-1
        self.env.step(command);self.state=self.gather();self.age+=1
        reward,success,failure,details=self.progress.update(self.state,action,self.skill)
        self.episode_return+=reward;self.episode_peak=torch.maximum(self.episode_peak,details['peak_force'])
        timeout=(self.age>=self.episode_steps)&~success&~failure
        done=timeout|success|failure
        terminal=observe(self.state,self.reference,self.progress).cpu().numpy()
        infos=[{} for _ in range(self.num_envs)]
        ids=done.nonzero().flatten()
        for i in ids.cpu().tolist():
            row=dict(skill=self.skill,success=bool(success[i]),failure=bool(failure[i]),timeout=bool(timeout[i]),
                length=int(self.age[i]),return_=float(self.episode_return[i]),
                grasp_psm1=bool(self.progress.flags[i,0]),lift_stable=bool(self.progress.flags[i,1]),
                grasp_psm2=bool(self.progress.flags[i,2]),handover=bool(self.progress.flags[i,3]),
                full_success=bool(details['full_success'][i]),hole_error_mm=float(details['hole_error'][i]*1000),
                peak_force_n=float(self.episode_peak[i]),assisted_grasp=False)
            self.episodes.append(row)
            infos[i]=dict(terminal_observation=terminal[i].copy(),TimeLimit_truncated=bool(timeout[i]),
                episode=dict(r=row['return_'],l=row['length']),metrics=row)
            infos[i]['TimeLimit.truncated']=bool(timeout[i])
        if len(ids):
            self.env.reset(env_ids=ids);self.state=self.gather()
            self.reference.reset(self.state,ids);self.progress.reset(ids,self.state)
            self.age[ids]=0;self.episode_return[ids]=0;self.episode_peak[ids]=0
            terminal=observe(self.state,self.reference,self.progress).cpu().numpy()
        return terminal,reward.cpu().numpy(),done.cpu().numpy(),infos

    def close(self):self.env.close()
    def get_attr(self,name,indices=None):return [getattr(self,name,None) for _ in self._get_indices(indices)]
    def set_attr(self,name,value,indices=None):setattr(self,name,value)
    def env_method(self,name,*args,indices=None,**kwargs):return [getattr(self,name)(*args,**kwargs) for _ in self._get_indices(indices)]
    def env_is_wrapped(self,wrapper_class,indices=None):return [False for _ in self._get_indices(indices)]
