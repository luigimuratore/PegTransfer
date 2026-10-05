"""Meaningful CPU regressions: sequencing, guards, replay and joint SAC/BC.

The ideal kinematics below are a test surrogate, not Isaac/PhysX validation.
Never infer real reachability, contact stability or policy performance from it.
"""
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest
import gymnasium as gym
import numpy as np
import torch
from common import SCHEMA, TASK, SURGICAL, assets, fingerprint

root=ModuleType('_sequence_cpu');root.__path__=[str(SURGICAL)];sys.modules[root.__name__]=root
for name in ('peg_transfer','peg_transfer_sequence'):
    package=ModuleType('_sequence_cpu.'+name);package.__path__=[str(SURGICAL/name)]
    sys.modules[package.__name__]=package
m=importlib.import_module('_sequence_cpu.peg_transfer_sequence.mdp')
physical=m.physical;g=m.g
assistance=m.assistance

class Object:
    def __init__(self,n):
        pose=torch.zeros(n,7);pose[:,3]=1
        self.data=NS(root_pose_w=pose,root_pos_w=pose[:,:3],root_quat_w=pose[:,3:],
            default_root_state=torch.cat([pose,torch.zeros(n,6)],dim=1),
            root_lin_vel_w=torch.zeros(n,3),root_ang_vel_w=torch.zeros(n,3))
    def write_root_pose_to_sim(self,pose,env_ids=None):
        self.data.root_pose_w[slice(None) if env_ids is None else env_ids]=pose
    def write_root_velocity_to_sim(self,velocity,env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self.data.root_lin_vel_w[ids]=velocity[:,:3];self.data.root_ang_vel_w[ids]=velocity[:,3:]

def fake(n=1):
    class Scene(dict): pass
    scene=Scene(object=Object(n));scene.env_origins=torch.zeros(n,3)
    for arm in (1,2):
        q=torch.zeros(n,4);q[:,0]=1
        joints=torch.zeros(n,8);joints[:,6:]=.5
        scene[f'robot_{arm}']=NS(data=NS(joint_pos=joints,default_joint_pos=joints.clone(),
            joint_vel=torch.zeros(n,8),root_quat_w=q),find_joints=lambda _:([6,7],[]))
        positions=torch.tensor([[-.069,0.,.077]] if arm==1 else [[.09,0.,.077]]).repeat(n,1)
        scene[f'ee_{arm}_frame']=NS(data=NS(target_pos_w=positions[:,None],target_quat_w=q[:,None].clone()))
    env=NS(scene=scene,num_envs=n,device='cpu',step_dt=.02,common_step_counter=0,
        cfg=NS(source_post='L5',phase='full',skill='full',guidance=1.,residual_translation=.0001,
               sequence_gamma=.999,stage_budget=600),reset_buf=torch.zeros(n,dtype=torch.bool),
        _sequence_action=torch.zeros(n,4),_sequence_pending={},
        action_manager=NS(get_term=lambda _:NS(cfg=NS(scale=(.006,.006,.006,.015,.015,.015)))))
    physical.reset(env);m.reset(env)
    return env

def tick(env, action):
    applied=m.motor_action(env,action)
    for arm,offset in ((1,0),(2,7)):
        ee=env.scene[f'ee_{arm}_frame'].data.target_pos_w[:,0]
        ee+=m.rotate_inverse(env.scene[f'robot_{arm}'].data.root_quat_w,applied[:,offset:offset+3]*.006)
        env.scene[f'robot_{arm}'].data.joint_pos[:,6:]=(.07+.43*(applied[:,offset+6,None]+1)/2)
    assistance.begin_step(env)
    for _ in range(4): assistance.assist(env)
    ps=physical.state(env)
    released=ps.holder==0
    p=physical.local_peg(env)
    p[released,2]=(p[released,2]-.002).clamp_min(g.REST_Z)
    env.scene['object'].data.root_pos_w[:]=p
    env.common_step_counter+=1
    physical.update(env);m.advance(env)

def held_env():
    env=fake();ps=physical.state(env)
    p=env.scene['object'].data.root_pos_w
    p[:]=torch.tensor(g.HANDOVER)
    env.scene['ee_1_frame'].data.target_pos_w[:,0]=p+torch.tensor(g.GRASP1)
    ps.offset[:]=-torch.tensor(g.GRASP1)
    ps.holder[:]=1;ps.source_cleared[:]=True;ps.lift_stable[:]=True
    env.scene['robot_1'].data.joint_pos[:,6:]=.07
    assistance.assist(env)
    return env


class AssistanceTests(unittest.TestCase):
    def test_sequence_step_captures_terminal_before_async_reset(self):
        # Exercise the actual sequence step override without importing Isaac.
        stub=ModuleType('_sequence_cpu.peg_transfer.sac_env')
        stub.PegTransferSACEnv=type('StubBase',(),{})
        sys.modules[stub.__name__]=stub
        cls=importlib.import_module('_sequence_cpu.peg_transfer_sequence.env').SequenceEnv
        env=cls.__new__(cls);env.__dict__.update(fake(2).__dict__)
        env.cfg.decimation=4;env.cfg.sim=NS(render_interval=4);env.cfg.rerender_on_reset=False
        env._sim_step_counter=0;env.physics_dt=.005
        env.episode_length_buf=torch.zeros(2,dtype=torch.long);env.extras={}
        env.sim=NS(has_gui=lambda:False,has_rtx_sensors=lambda:False,step=lambda **_:None)
        env.scene.write_data_to_sim=lambda:None;env.scene.update=lambda **_:None
        env.action_manager.process_action=lambda action:None;env.action_manager.apply_action=lambda:None
        env.recorder_manager=NS(active_terms=[],record_pre_step=lambda:None,
            record_post_physics_decimation_step=lambda:None,record_pre_reset=lambda _:None,
            record_post_reset=lambda _:None)
        env.termination_manager=NS(compute=lambda:torch.tensor([True,False]),
            terminated=torch.tensor([False,False]),time_outs=torch.tensor([True,False]),
            get_term=lambda _:torch.tensor([False,False]))
        env.reward_manager=NS(compute=lambda dt:m.reward(env)*dt)
        env.observation_manager=NS(compute=lambda **_:dict(policy=m.observation(env)))
        env.command_manager=NS(compute=lambda **_:None)
        original=physical.local_peg(env).clone()
        def reset_one(ids):
            physical.reset(env,ids);m.reset(env,ids)
            env.scene['object'].data.root_pos_w[ids]=torch.tensor([.123,.01,.02])
        env._reset_idx=reset_one
        obs,_,term,trunc,_=env.step(torch.zeros(2,4))
        self.assertFalse(term.any());self.assertTrue(trunc[0])
        np.testing.assert_allclose(env._sac_terminal[0]['observation'][9:12],original[0].numpy()*100)
        np.testing.assert_allclose(env._sac_terminal[0]['sequence']['peg_root_m'],original[0].numpy())
        np.testing.assert_allclose(obs['policy'][0,9:12],np.array([12.3,1.,2.]),rtol=1e-6)
        torch.testing.assert_close(physical.local_peg(env)[1],original[1])

    def test_contact_tilt_reproduces_old_freeze_and_new_servo_keeps_hold(self):
        env=held_env();obj=env.scene['object'];ps=physical.state(env)
        ref=obj.data.root_pos_w.clone()
        obj.data.root_quat_w[:,1]=.007;obj.data.root_quat_w[:,0]=(1-.007**2)**.5
        obj.data.root_pos_w[:,2]-=.00065
        obj.data.root_lin_vel_w[:,2]=-.2
        physical.assist(env)
        self.assertLess(float(obj.data.root_pos_w[0,2]),float(ref[0,2]))
        self.assertLess(float(obj.data.root_lin_vel_w[0,2]),0.)  # Old assist skipped this held pose.
        assistance.assist(env)
        torch.testing.assert_close(obj.data.root_pos_w,ref)
        torch.testing.assert_close(obj.data.root_quat_w,torch.tensor([[1.,0.,0.,0.]]))
        self.assertEqual(int(ps.holder[0]),1)
        self.assertEqual(int(ps.contact_corrections[0]),1)
        self.assertEqual(float(obj.data.root_lin_vel_w.norm()),0.)

    def test_initial_capture_threshold_is_not_relaxed(self):
        env=fake();obj=env.scene['object']
        env.scene['ee_1_frame'].data.target_pos_w[:,0]=physical.local_peg(env)+torch.tensor(g.GRASP1)
        env.scene['robot_1'].data.joint_pos[:,6:]=.07
        obj.data.root_quat_w[:,1]=.007;obj.data.root_quat_w[:,0]=(1-.007**2)**.5
        for _ in range(12):assistance.assist(env)
        self.assertEqual(int(physical.state(env).holder[0]),0)
        self.assertGreater(float(obj.data.root_quat_w[0,1]),.005)

    def test_contact_displacement_cannot_fake_receiver_capture(self):
        env=held_env();ps=physical.state(env);obj=env.scene['object']
        ref=obj.data.root_pos_w.clone()
        env.scene['robot_2'].data.joint_pos[:,6:]=.07
        env.scene['ee_2_frame'].data.target_pos_w[:,0]=ref+torch.tensor(g.GRASP2)+torch.tensor([0.,0.,-.005])
        for _ in range(12):
            obj.data.root_pos_w[:]=ref+torch.tensor([0.,0.,-.003])
            assistance.assist(env)
        self.assertFalse(ps.receiver_latched.any())
        self.assertEqual(int(ps.candidate2[0]),0)

    def test_unsafe_post_motion_stays_held_at_last_safe_pose(self):
        env=held_env();ps=physical.state(env);obj=env.scene['object']
        ref=torch.tensor([[g.TARGET[0]+.0025,g.TARGET[1],g.CLEAR_Z]])
        obj.data.root_pos_w[:]=ref;ps.held_reference[:]=ref
        tool=env.scene['ee_1_frame'].data.target_pos_w[:,0]
        tool[:]=ref-ps.offset;tool[:,2]-=.002
        assistance.assist(env)
        torch.testing.assert_close(obj.data.root_pos_w,ref)
        self.assertEqual(int(ps.holder[0]),1)
        self.assertEqual(int(ps.blocked_moves[0]),1)
        tool[:,2]-=.008
        assistance.assist(env)
        self.assertEqual(int(ps.holder[0]),0)
        self.assertEqual(assistance.telemetry(env)['break_reason'],['tool_stretch'])

    def test_large_contact_disturbance_releases_instead_of_erasing_it(self):
        for mode in ('rotation','translation'):
            env=held_env();obj=env.scene['object']
            if mode=='rotation':obj.data.root_quat_w[:,1]=.05
            else:obj.data.root_pos_w[:,2]-=.005
            obj.data.root_lin_vel_w[:,2]=-.2
            before=obj.data.root_pose_w.clone()
            assistance.assist(env)
            self.assertEqual(int(physical.state(env).holder[0]),0)
            torch.testing.assert_close(obj.data.root_pose_w,before)
            self.assertLess(float(obj.data.root_lin_vel_w[0,2]),0.)

    def test_release_restores_free_dynamics_and_initial_reset_clears_reference(self):
        env=held_env();obj=env.scene['object'];ps=physical.state(env)
        env.scene['robot_1'].data.joint_pos[:,6:]=.5
        obj.data.root_lin_vel_w[:,2]=-.2
        assistance.assist(env)
        self.assertEqual(int(ps.holder[0]),0)
        self.assertFalse(ps.reference_valid.any())
        self.assertLess(float(obj.data.root_lin_vel_w[0,2]),0.)
        physical.reset(env)
        self.assertFalse(ps.reference_valid.any())
        self.assertEqual(int(ps.contact_corrections[0]),0)


class SequenceTests(unittest.TestCase):
    def test_receiver_open_approach_stays_outside_opposite_y_edge(self):
        env=held_env();s=m.state(env)
        for stage in (8,9,10):
            s.stage[:]=stage;s.entered[:]=True
            goal=m.goals(env)[0,1]
            root=physical.local_peg(env)[0]
            self.assertGreater(float(goal[1]-root[1]),g.PEG_MAX[1])
            self.assertAlmostEqual(float(goal[0]-root[0]),g.GRASP2[0],places=6)
            self.assertEqual(float(m.jaw_targets(env)[0,1]),1.)
        distance=(goal-root-goal.new_tensor(g.GRASP2)).norm()
        self.assertLess(float(distance),g.GRASP_RADIUS)
        self.assertGreater(g.GRASP2[1],0.)
        self.assertLess(g.GRASP1[1],0.)

    def test_full_measured_sequence_under_ideal_kinematics(self):
        env=fake(); visited=[]
        for _ in range(3000):
            stage=int(m.state(env).stage[0])
            if not visited or visited[-1]!=stage: visited.append(stage)
            if stage==20: break
            tick(env,m.expert_action(env))
            self.assertFalse(m.failure(env).any(),f'stage {stage}, age {m.state(env).age.tolist()}')
        self.assertEqual(visited,list(range(21)))
        self.assertTrue(m.success(env).all())
        self.assertTrue(all(getattr(physical.state(env),key).all() for key in physical.METRICS))

    def test_position_or_elapsed_time_cannot_fake_transfer(self):
        env=fake()
        env.scene['object'].data.root_pos_w[:]=torch.tensor(g.TARGET)
        m.state(env).stage[:]=11
        for _ in range(30):
            env.common_step_counter+=1;m.advance(env)
        self.assertEqual(int(m.state(env).stage[0]),11)
        self.assertFalse(m.success(env).any())
        self.assertFalse(physical.state(env).full_success.any())

    def test_receiver_cannot_advance_on_distance_without_open_jaw_and_donor(self):
        env=fake();s=m.state(env);s.stage[:]=10
        p=physical.local_peg(env)
        env.scene['ee_2_frame'].data.target_pos_w[:,0]=p+torch.tensor(g.GRASP2)
        env.scene['robot_2'].data.joint_pos[:,6:]=.07
        env.common_step_counter+=1;m.advance(env)
        self.assertEqual(int(s.stage[0]),10)

    def test_historical_full_success_cannot_hide_a_disturbed_placement(self):
        env=fake();m.state(env).stage[:]=19
        ps=physical.state(env);ps.full_success[:]=True;ps.settled_steps[:]=0
        env.common_step_counter+=1;m.advance(env)
        self.assertEqual(int(m.state(env).stage[0]),19)
        self.assertFalse(m.success(env).any())

    def test_no_guidance_means_no_active_translation_from_teacher(self):
        env=fake();env.cfg.guidance=0.;s=m.state(env);s.stage[:]=1
        result=m.motor_action(env,torch.zeros(1,4))
        torch.testing.assert_close(result[:,:3],torch.zeros(1,3))
        action=torch.tensor([[.2,-.1,.3,-.8]])
        result=m.motor_action(env,action)
        torch.testing.assert_close(result[:,:3]*.006,action[:,:3]*.0012)
        self.assertAlmostEqual(float(result[0,6]),-.8,places=6)

    def test_expert_has_identical_physical_effect_at_every_guidance(self):
        env=fake();m.state(env).stage[:]=1
        reference=None
        for alpha in (1.,.75,.5,.25,0.):
            env.cfg.guidance=alpha
            actual=m.motor_action(env,m.expert_action(env))
            if reference is None: reference=actual
            else: torch.testing.assert_close(actual,reference,atol=1e-6,rtol=1e-5)
        self.assertEqual(m.observation(env).shape,(1,112))

    def test_async_reset_does_not_erase_other_sequence(self):
        env=fake(2);s=m.state(env);s.stage[:]=14;s.potential[:]=22
        m.reset(env,torch.tensor([0]))
        self.assertEqual(s.stage.tolist(),[0,14]);self.assertEqual(s.potential.tolist(),[0,22])

    def test_repeated_close_has_no_repeated_stage_bonus(self):
        env=fake();s=m.state(env);s.stage[:]=4
        physical.state(env).holder[:]=1
        env.common_step_counter+=1;m.advance(env)
        self.assertEqual(int(s.stage[0]),5)
        self.assertEqual(float(s.stage_event[0]),1.)
        env.common_step_counter+=1;m.advance(env)
        self.assertEqual(float(s.stage_event[0]),0.)

    def test_timeout_bootstraps_potential_but_true_failure_clears_it(self):
        env=fake();s=m.state(env);s.stage[:]=1
        env.reset_buf[:]=True  # Time-limit truncation, no physical failure.
        m.reward(env)
        self.assertGreater(float(s.potential[0]),2.)
        s.age[:]=env.cfg.stage_budget
        env.common_step_counter+=1
        m.reward(env)
        self.assertEqual(float(s.potential[0]),0.)
        self.assertIn('stage_budget_exhausted',env._sequence_pending[0]['failure_causes'])

class FakeAutoReset:
    def __init__(self,dim=8):
        self.unwrapped=self;self.num_envs=2;self.device='cpu';self.render_mode=None
        self.dim=dim
        self.single_observation_space={'policy':gym.spaces.Box(-np.inf,np.inf,(dim,),dtype=np.float32)}
        self.single_action_space=gym.spaces.Box(-1,1,(4,),dtype=np.float32)
    def reset(self): return {'policy':torch.zeros(2,self.dim)},{}
    def step(self,action):
        self._sac_terminal={i:dict(observation=np.full(self.dim,11+i,np.float32),metrics=dict(full_success=False),
            phase_success=False,sequence=dict(stage=1),failure=i==0,blocked_moves=0,source='L5') for i in range(2)}
        return {'policy':torch.full((2,self.dim),999.)},torch.ones(2),torch.tensor([True,False]),torch.tensor([False,True]),{}
    def close(self): pass

def archive(directory,dim=8):
    path=Path(directory)/'test.npz'
    obs=np.arange(4*dim,dtype=np.float32).reshape(4,dim)/(4*dim)
    next_obs=np.concatenate([obs[1:],np.zeros((1,dim),np.float32)])
    np.savez_compressed(path,observations=obs,next_observations=next_obs,
        actions=np.zeros((4,4),np.float32),bc_actions=np.tile([.1,-.1,.1,.8],(4,1)).astype(np.float32),
        rewards=np.zeros(4,np.float32),dones=np.array([0,0,0,1],np.float32),
        stages=np.array([1,1,2,2]),guidance=np.zeros(4,np.float32))
    import hashlib
    meta=dict(schema=SCHEMA+'-demonstrations',task=TASK,fingerprint=fingerprint(),assets=assets(),
        transitions=4,observation_dim=dim,episodes=[dict(start=0,stop=4,full_success=True)],
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.with_suffix('.json').write_text(json.dumps(meta))
    return path

def observation_space_env():
    class Minimal(gym.Env):
        observation_space=gym.spaces.Box(-np.inf,np.inf,(112,),dtype=np.float32)
        action_space=gym.spaces.Box(-1,1,(4,),dtype=np.float32)
        def reset(self,seed=None,options=None): return np.zeros(112,np.float32),{}
        def step(self,action): return np.zeros(112,np.float32),0.,False,False,{}
    return Minimal()

class LearningTests(unittest.TestCase):
    def test_goal_actor_joint_sac_bc_on_real_observation_dimensions(self):
        from adapter import SequenceVecEnv
        from model import AnchoredSAC,DemoReplay,load_demos
        from policy import GoalPolicy
        env=SequenceVecEnv(FakeAutoReset(112))
        with tempfile.TemporaryDirectory() as directory:
            path=archive(directory,112);data,_=load_demos(path)
            model=AnchoredSAC(GoalPolicy,env,device='cpu',seed=42,buffer_size=32,
                learning_starts=0,batch_size=8,gradient_steps=1,ent_coef=.001,
                replay_buffer_class=DemoReplay,replay_buffer_kwargs=dict(demo_path=str(path)),
                policy_kwargs=dict(net_arch=[16,16]))
            model.bc_weight=5.;model._demo_data=data
            model.learn(4)
            self.assertGreater(model._n_updates,0)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.policy.parameters()))
            self.assertEqual(model.actor.features_extractor.features_dim,29)
            self.assertEqual(model.critic.features_extractor.features_dim,112)

    def test_latest_artifacts_ignore_failed_or_incompatible_results(self):
        import paths
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'logs/sac_sequence';folder.mkdir(parents=True)
            valid=dict(task=TASK,fingerprint=fingerprint(),assets=assets(),source='L5',target='R2',
                baseline_controller=True,skill='full',complete=True,episodes=3,full_success_rate=1.)
            good=folder/'baseline_full_good.json';good.write_text(json.dumps(valid))
            (folder/'baseline_full_failed.json').write_text(json.dumps(dict(valid,full_success_rate=0.)))
            with patch.object(paths,'REPO',Path(directory)):
                self.assertEqual(paths.latest('baseline'),good)
                with self.assertRaises(SystemExit):paths.latest('checkpoint')

    def test_actor_is_independent_of_clock_and_previous_reward_state(self):
        from stable_baselines3 import SAC
        from policy import GoalPolicy
        model=SAC(GoalPolicy,observation_space_env(),device='cpu',buffer_size=32,
                  policy_kwargs=dict(net_arch=[16,16]),seed=42)
        obs=np.zeros((1,112),np.float32)
        before=model.predict(obs,deterministic=True)[0]
        obs[:,108:]=10  # Skill cutoff, stage age, dwell and potential are critic-only.
        obs[:,28:60]=2  # Joint state also remains critic-only; error is already body-frame.
        np.testing.assert_array_equal(before,model.predict(obs,deterministic=True)[0])
        self.assertEqual(model.critic.features_extractor.features_dim,112)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'goal_policy.zip';model.save(path)
            native=SAC.load(path,device='cpu')
            np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],native.predict(obs,deterministic=True)[0])
    def test_terminal_replay_and_joint_sac_bc_save_load(self):
        from adapter import SequenceVecEnv
        from model import AnchoredSAC,DemoReplay,load_demos,pretrain
        from stable_baselines3 import SAC
        env=SequenceVecEnv(FakeAutoReset())
        _,_,_,info=env.step(np.zeros((2,4),np.float32))
        np.testing.assert_array_equal(info[1]['terminal_observation'],np.full(8,12))
        self.assertTrue(info[1]['TimeLimit.truncated']);self.assertFalse(info[0]['TimeLimit.truncated'])
        with tempfile.TemporaryDirectory() as directory:
            path=archive(directory);data,_=load_demos(path)
            model=AnchoredSAC('MlpPolicy',env,device='cpu',seed=42,verbose=0,buffer_size=32,
                learning_starts=0,batch_size=8,gradient_steps=1,ent_coef=.001,
                replay_buffer_class=DemoReplay,replay_buffer_kwargs=dict(demo_path=str(path),demo_fraction=.5),
                policy_kwargs=dict(net_arch=[16,16]))
            model.bc_weight=5.;model._demo_data=data
            pretrain(model,data,4,batch_size=8)
            model.learn(8)
            self.assertGreater(model._n_updates,0)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.policy.parameters()))
            model.save(Path(directory)/'sac.zip')
            reloaded=SAC.load(Path(directory)/'sac.zip',device='cpu')
            obs=np.zeros((2,8),np.float32)
            np.testing.assert_allclose(model.predict(obs,deterministic=True)[0],reloaded.predict(obs,deterministic=True)[0])
            sampled=model.replay_buffer.sample(8)
            online=sampled.observations[:,0]>100
            self.assertTrue((sampled.next_observations[online,0]>=11).all())

    def test_demo_integrity_and_curriculum_gate(self):
        from model import load_demos
        from curriculum import passed
        with tempfile.TemporaryDirectory() as directory:
            path=archive(directory)
            meta=json.loads(path.with_suffix('.json').read_text());meta['episodes'][0]['full_success']=False
            path.with_suffix('.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'successful'):load_demos(path)
        self.assertFalse(passed(dict(complete=False,episodes=100,baseline_controller=False,skill_success_rate=1.),.8,100))
        self.assertFalse(passed(dict(complete=True,episodes=100,baseline_controller=True,skill_success_rate=1.),.8,100))
        self.assertTrue(passed(dict(complete=True,episodes=100,baseline_controller=False,skill_success_rate=.9),.8,100))

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
