"""Meaningful CUDA checks for free-action SAC and measured task success."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from stable_baselines3.common.logger import configure
from rl_core import *
from rl_data import trace_state
from rl_learning import *

ROOT=Path(__file__).resolve().parents[2]

class FreeSACChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise RuntimeError('Run these checks with GPU access')
        torch.set_num_threads(4);cls.device='cuda:0'
        cls.profile=json.loads((ROOT/'tools/peg_grasp_v2/peg_profile.json').read_text())
        with (ROOT/'logs/grasp_v2/full_assisted_validation3_2026-10-05.trace.jsonl').open() as f:
            cls.row=json.loads(next(f))
        cls.data,cls.meta=load_data(ROOT/'logs/grasp_v2/sac_free_demonstrations_v3.npz',cls.device)

    def state(self,n=1):
        s=trace_state(self.row['pre_action_state'],None,self.device)
        return {k:v.expand(n,*v.shape[1:]).clone() for k,v in s.items()}

    def test_async_reset_preserves_other_progress_and_reference(self):
        s=self.state(2);p=Progress(2,self.device,self.profile);r=MotorReference(s)
        p.flags[:]=True;p.contact_age[:]=1.;r.p+=.001
        p.reset(torch.tensor([0],device=self.device),s);r.reset(s,torch.tensor([0],device=self.device))
        self.assertFalse(bool(p.flags[0].any()));self.assertTrue(bool(p.flags[1].all()))
        torch.testing.assert_close(r.p[1],s['tip'][1]+.001)

    def test_no_contact_or_historical_position_can_fake_lift_or_handover(self):
        s=self.state();p=Progress(1,self.device,self.profile);p.reset(torch.tensor([0],device=self.device),s)
        s['bottom'][:]=.06
        for _ in range(250):p.update(s,torch.zeros(1,14,device=self.device))
        self.assertFalse(bool(p.flags.any()))
        p.flags[0,:3]=True;p.update(s,torch.zeros(1,14,device=self.device))
        self.assertFalse(bool(p.flags[0,3]))

    def test_free_placement_requires_release_hole_and_stability(self):
        s=self.state();p=Progress(1,self.device,self.profile)
        s['peg'][:]=torch.tensor([.042-self.profile['hole_center_m'][0],.015,.010-self.profile['bounds_m'][0][2]],device=self.device)
        s['peg_q'][:]=torch.tensor([1.,0.,0.,0.],device=self.device);s['bottom'][:]=.010;s['speed'][:]=0
        p.reset(torch.tensor([0],device=self.device),s);p.flags[:]=True
        s['forces'][0,1,0,0]=.2
        for _ in range(110):_,_,_,details=p.update(s,torch.zeros(1,14,device=self.device))
        self.assertFalse(bool(details['full_success'][0]))
        s['forces'].zero_()
        for _ in range(110):_,success,_,details=p.update(s,torch.zeros(1,14,device=self.device))
        self.assertTrue(bool(success[0]));self.assertGreater(float(details['clearance'][0]),.0001)
        s['peg'][:,0]+=.02
        _,success,_,_=p.update(s,torch.zeros(1,14,device=self.device))
        self.assertFalse(bool(success[0]))

    def test_motor_actions_are_signed_and_control_both_arms(self):
        s=self.state(2);r=MotorReference(s);a=torch.zeros(2,14,device=self.device)
        a[0,0]=1;a[1,0]=-1;a[:,10]=1;a[:,13]=1;r.advance(a,s)
        self.assertGreater(float(r.p[0,0,0]-s['tip'][0,0,0]),0)
        self.assertLess(float(r.p[1,0,0]-s['tip'][1,0,0]),0)
        self.assertGreater(float(rotvec(multiply(r.q[0,1],conjugate(s['tool_q'][0,1]))).norm()),0)
        self.assertGreater(float(r.jaw[0,1]),.15)

    def test_repeated_closure_has_one_grasp_event(self):
        s=self.state();p=Progress(1,self.device,self.profile);p.reset(torch.tensor([0],device=self.device),s)
        s['forces'][0,0,0]=torch.tensor([.2,0.,0.],device=self.device)
        s['forces'][0,0,1]=torch.tensor([-.2,0.,0.],device=self.device);s['contact_valid'][0,0]=True
        events=0
        for _ in range(120):
            before=p.flags.clone();p.update(s,torch.zeros(1,14,device=self.device))
            events+=int(bool(p.flags[0,0]&~before[0,0]))
        self.assertEqual(events,1)

    def test_terminal_replay_bootstraps_timeout_without_reset_observation(self):
        obs=Box(-np.inf,np.inf,(OBS_DIM,),dtype=np.float32);act=Box(-1,1,(14,),dtype=np.float32)
        buffer=CUDAReplayBuffer(4,obs,act,n_envs=2,device=self.device,demo_fraction=0)
        current=np.zeros((2,OBS_DIM),np.float32);terminal=np.ones_like(current)*7
        buffer.add(current,terminal,np.zeros((2,14)),np.zeros(2),np.ones(2),[{'TimeLimit.truncated':True},{}])
        torch.testing.assert_close(buffer.done[0,:,0],torch.tensor([0.,1.],device=self.device))
        torch.testing.assert_close(buffer.nxt[0],torch.full((2,OBS_DIM),7.,device=self.device))

    def test_real_dimensions_sac_actor_critic_bc_and_save_load(self):
        class SpaceEnv(gym.Env):
            observation_space=Box(-np.inf,np.inf,(OBS_DIM,),dtype=np.float32)
            action_space=Box(-1,1,(ACTION_DIM,),dtype=np.float32)
            def reset(self,**kwargs):return np.zeros(OBS_DIM,dtype=np.float32),{}
            def step(self,a):return np.zeros(OBS_DIM,dtype=np.float32),0.,False,False,{}
        env=DummyVecEnv([SpaceEnv])
        model=FreeSAC('MlpPolicy',env,device=self.device,learning_rate=1e-4,buffer_size=100,
            learning_starts=0,batch_size=64,ent_coef=1e-5,replay_buffer_class=CUDAReplayBuffer,
            replay_buffer_kwargs=dict(demonstrations=str(ROOT/'logs/grasp_v2/sac_free_demonstrations_v3.npz'),skill='lift'),
            policy_kwargs=dict(net_arch=[256,256],features_extractor_class=StandardFeatures,
                features_extractor_kwargs=dict(mean=self.meta['mean'],std=self.meta['std'])))
        model.set_logger(configure(None,[]));model._total_timesteps=100
        initialize_actor(model,self.data,self.meta,updates=2,batch_size=64)
        actor=[p.detach().clone() for p in model.actor.parameters()]
        critic=[p.detach().clone() for p in model.critic.parameters()]
        model.train(2,64)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(actor,model.actor.parameters())))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(critic,model.critic.parameters())))
        self.assertTrue(all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()))
        x=self.data['obs'][:8].cpu().numpy();before=model.predict(x,deterministic=True)[0]
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            model.save(Path(folder)/'policy.zip');loaded=FreeSAC.load(Path(folder)/'policy.zip',device=self.device)
            np.testing.assert_allclose(before,loaded.predict(x,deterministic=True)[0],atol=1e-6)
        env.close()

if __name__=='__main__':unittest.main(verbosity=2)
