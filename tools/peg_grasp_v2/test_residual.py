"""CUDA checks for residual bounds, asynchronous resets and SAC replay/update."""
from pathlib import Path
import sys
import unittest
import tempfile
import json
from types import SimpleNamespace
import numpy as np
import torch
from gymnasium.spaces import Box
from stable_baselines3.common.vec_env import VecEnv

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.checkpoint_runs/checkpoint13-three-posts-fast-assisted/model/sources'))
from residual_env import ResidualEnv, ResidualLimits, mix_actions, OBS_DIM
from residual_learning import make_model, ResidualSAC, warm_critic
from rl_core import observe


class ToyBC(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight=torch.nn.Parameter(torch.ones(1,device='cuda'))
        self.selected_option=self.selected_goal=self.selected_prediction=self.current_goal=self.decision_scores=None
    def reset(self,ids=None):
        if ids is None:self.selected_option=self.selected_goal=self.selected_prediction=self.current_goal=None
        elif self.selected_option is not None:self.selected_option[ids]=-1
    def forward(self,x):
        n=len(x)
        if self.selected_option is None:
            self.selected_option=torch.zeros(n,device=x.device,dtype=torch.long)
            self.selected_prediction=torch.zeros(n,16,device=x.device)
        fresh=self.selected_option<0
        self.selected_option=torch.zeros_like(self.selected_option)
        self.selected_prediction=torch.where(fresh[:,None],0.,self.selected_prediction)+1
        self.selected_goal=self.current_goal=self.selected_prediction.clone()
        self.decision_scores=torch.zeros(n,19,device=x.device)
        return torch.zeros(n,14,device=x.device)


class ToyEnv(VecEnv):
    def __init__(self):
        super().__init__(2,Box(-np.inf,np.inf,(110,),np.float32),Box(-1,1,(14,),np.float32))
        self.device='cuda:0'; self.episode_steps=3
        z=lambda *shape:torch.zeros(*shape,device='cuda')
        q=lambda *shape:torch.nn.functional.pad(torch.ones(*shape,1,device='cuda'),(0,3))
        self.state=dict(tip=z(2,2,3),tool_q=q(2,2),joints=z(2,2,6),joint_vel=z(2,2,6),jaws=z(2,2,2),
            forces=z(2,2,2,3),relative=z(2,2,3),relative_q=q(2,2),peg=z(2,3),peg_q=q(2),linvel=z(2,3),angvel=z(2,3),
            bottom=z(2),target=z(2,3),source=z(2,3))
        self.reference=SimpleNamespace(p=z(2,2,3),q=q(2,2),jaw=z(2,2),grip_age=z(2,2),speed_cap=z(2,2),anchor=z(2,2,3),anchor_q=q(2,2))
        self.progress=SimpleNamespace(flags=z(2,4).bool(),contact_age=z(2,2),lift_age=z(2),place_age=z(2),receiver_alone_age=z(2),
                                     anchor=z(2,2,3),anchor_q=q(2,2),update=self.update)
        self.support=SimpleNamespace(holder=z(2).long(),relative=z(2,3),relative_q=q(2),acquire_jaw=z(2),placement_active=z(2).bool())
        self.profile=dict(bounds_m=[[-.01,-.01,-.01],[.01,.01,.01]])
        self.reset_records=[dict(pair='a'),dict(pair='b')];self.age=z(2).long()
    def update(self,state,action,skill):
        return torch.zeros(2,device='cuda'),self.age>=3,torch.zeros(2,device='cuda',dtype=torch.bool),{}
    def reset(self):
        self.age.zero_();return observe(self.state,self.reference,self.progress).cpu().numpy()
    def step_async(self,actions):self.pending=actions
    def step_wait(self):
        self.age+=torch.tensor([1,0],device='cuda')
        self.state['peg'][:,0]+=1
        reward,success,failure,_=self.progress.update(self.state,torch.tensor(self.pending,device='cuda'),'full')
        dones=(success|failure).cpu().numpy();infos=[{},{}]
        if dones[0]:
            infos[0]=dict(metrics=dict(success=True),**{'TimeLimit.truncated':True})
            self.state['peg'][0,0]=-1;self.age[0]=0
        return observe(self.state,self.reference,self.progress).cpu().numpy(),reward.cpu().numpy(),dones,infos
    def close(self):pass
    def get_attr(self,name,indices=None):return [getattr(self,name,None)]*len(self._get_indices(indices))
    def set_attr(self,name,value,indices=None):setattr(self,name,value)
    def env_method(self,name,*args,indices=None,**kw):return []
    def env_is_wrapped(self,wrapper_class,indices=None):return [False]*len(self._get_indices(indices))


class ResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise RuntimeError('These tests require the RTX GPU')
        torch.set_num_threads(8)
    def test_zero_residual_and_vector_bounds_keep_interactions(self):
        base=torch.rand(20,14,device='cuda')*1.8-.9
        mixed,_=mix_actions(base,torch.zeros_like(base),torch.ones(20,2,device='cuda'),ResidualLimits())
        torch.testing.assert_close(base,mixed,rtol=0,atol=0)
        mixed,delta= mix_actions(base,torch.ones_like(base),torch.ones(20,2,device='cuda'),ResidualLimits())
        self.assertLessEqual(float(delta.reshape(-1,2,7)[...,:3].norm(dim=-1).max()),.050001)
        self.assertTrue(torch.equal(base.reshape(-1,2,7)[...,3:],mixed.reshape(-1,2,7)[...,3:]))
        mixed,_=mix_actions(base,torch.ones_like(base),torch.zeros(20,2,device='cuda'),ResidualLimits())
        self.assertTrue(torch.equal(base,mixed))
    def test_async_reset_and_pre_reset_terminal_context(self):
        vec=ResidualEnv(ToyEnv(),ToyBC(),SimpleNamespace(apply=lambda a,s,r:a))
        obs=vec.reset(); self.assertEqual(obs.shape,(2,OBS_DIM))
        for _ in range(3):obs,rew,done,info=vec.step(np.zeros((2,14),np.float32))
        self.assertTrue(done[0]);self.assertFalse(done[1])
        # Terminal contains peg x=3; reset contains x=-1 (normalized /0.1).
        self.assertEqual(info[0]['terminal_observation'][82],20.)
        self.assertEqual(obs[0,82],-10.)
        self.assertEqual(float(vec.base.selected_prediction[0,0]),1.)
        self.assertEqual(float(vec.base.selected_prediction[1,0]),4.)
        self.assertEqual(info[0]['terminal_observation'][143],4.)
    def test_cuda_sac_timeout_update_and_save_load(self):
        vec=ResidualEnv(ToyEnv(),ToyBC(),SimpleNamespace(apply=lambda a,s,r:a))
        model=make_model(vec,buffer_size=128,batch_size=16)
        obs=vec.reset();self.assertTrue(np.equal(model.predict(obs,deterministic=True)[0],0).all())
        frozen={k:v.clone() for k,v in vec.base.state_dict().items()}
        for _ in range(8):
            action=np.zeros((2,14),np.float32);nxt,rew,done,info=vec.step(action);terminal=nxt.copy()
            for i in np.flatnonzero(done):terminal[i]=info[i]['terminal_observation']
            model.replay_buffer.add(obs,terminal,action,rew,done,info);obs=nxt
        self.assertTrue(model.replay_buffer.obs.is_cuda)
        self.assertEqual(float(model.replay_buffer.done.sum()),0.)
        critic_before=[p.clone() for p in model.critic.parameters()]
        warm_critic(model,4);model.train(4,16)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(critic_before,model.critic.parameters())))
        self.assertTrue(all(torch.equal(value,vec.base.state_dict()[key]) for key,value in frozen.items()))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'candidate.zip';model.save(path)
            loaded=ResidualSAC.load(path,env=vec,device='cuda:0')
            np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],loaded.predict(obs,deterministic=True)[0])

    def test_real_checkpoint13_zero_residual_and_frozen_weights(self):
        from autonomy_learning import load_policy,physical_observation
        path=ROOT/'.checkpoint_runs/checkpoint13-three-posts-fast-assisted/model/policy.pt'
        left=load_policy(path,'cuda:0');right=load_policy(path,'cuda:0')
        raw=ToyEnv();vec=ResidualEnv(raw,left,SimpleNamespace(apply=lambda a,s,r:a))
        obs=vec.reset()
        with torch.no_grad():expected=right(physical_observation(torch.as_tensor(raw.reset(),device='cuda')))
        torch.testing.assert_close(vec.next_base,expected,rtol=0,atol=0)
        self.assertTrue(all(not p.requires_grad for p in left.parameters()))
        self.assertEqual(obs.shape[1],249)
        mixed,_=mix_actions(vec.next_base,torch.zeros_like(expected),torch.ones(2,2,device='cuda'),ResidualLimits())
        self.assertTrue(torch.equal(expected,mixed))

    def test_kit_zero_exit_cannot_bypass_training_failure(self):
        from residual_pipeline import validate_child_result
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            (path/'status.json').write_text(json.dumps(dict(complete=True,stage='baseline_gate_failed')))
            with self.assertRaisesRegex(RuntimeError,'before producing a candidate'):
                validate_child_result(path,'train')
            (path/'error.json').write_text(json.dumps(dict(error='warm rollout timeout')))
            with self.assertRaisesRegex(RuntimeError,'warm rollout timeout'):
                validate_child_result(path,'train')

    def test_completed_status_cannot_hide_partial_evaluation(self):
        from residual_pipeline import validate_child_result
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            (path/'status.json').write_text(json.dumps(dict(complete=True,stage='evaluation_complete')))
            (path/'evaluation.json').write_text(json.dumps(dict(complete=False,episodes=11,requested_episodes=12)))
            with self.assertRaisesRegex(RuntimeError,'evaluation incomplete'):
                validate_child_result(path,'evaluate')

    def test_held_aperture_guard_recovers_without_overriding_release(self):
        from residual_grasp_guard import HeldJawGuard
        guard=HeldJawGuard(2,'cuda:0')
        forces=torch.zeros(2,2,2,3,device='cuda');forces[:,0,0,0]=.4;forces[:,0,1,0]=-.4
        state=dict(forces=forces,contact_valid=torch.ones(2,2,2,device='cuda',dtype=torch.bool))
        reference=SimpleNamespace(jaw=torch.full((2,2),.13,device='cuda'))
        support=SimpleNamespace(holder=torch.tensor([1,1],device='cuda'))
        base=torch.zeros(2,14,device='cuda');base[:,6]=4*.13-1
        guard.apply(base,state,reference,support)
        self.assertTrue(guard.valid[:,0].all())
        forces.mul_(.1);reference.jaw[:,0]=.12;base[:,6]=4*.09-1
        corrected=guard.apply(base,state,reference,support)
        self.assertAlmostEqual(float((corrected[0,6]+1)*.25),.1205,places=6)
        # Actual neural opening and an unheld arm are never overridden.
        opening=base.clone();opening[:,6]=4*.14-1
        self.assertTrue(torch.equal(guard.apply(opening,state,reference,support),opening))
        support.holder.zero_()
        self.assertTrue(torch.equal(guard.apply(base,state,reference,support),base))
        guard.reset(torch.tensor([0],device='cuda'))
        self.assertFalse(bool(guard.valid[0].any()));self.assertTrue(bool(guard.valid[1,0]))

    def test_donor_spring_retains_receiver_compliance_and_wrench_bounds(self):
        from residual_spring_support import install
        shared=dict(stiffness=300.,force_limit_n=.4,torque_limit_nm=.004)
        seen=[]
        support=SimpleNamespace(holder=torch.tensor([1,2,0],device='cuda'),settings=shared)
        support.reset=lambda ids:None
        support.apply=lambda:seen.append(support.settings['stiffness'].clone())
        support.evidence=lambda index=None:dict(stiffness_n_m=support.settings['stiffness'])
        install(support,900);support.apply()
        torch.testing.assert_close(seen[0].flatten(),torch.tensor([900.,300.,300.],device='cuda'))
        self.assertEqual(shared['stiffness'],300.)
        self.assertEqual(support.settings['stiffness'],900.)
        self.assertEqual(support.settings['force_limit_n'],.4)
        self.assertEqual(support.settings['torque_limit_nm'],.004)
        self.assertEqual(support.evidence()['stiffness_by_holder_n_m'],dict(psm1=900.,psm2=300.))

    def test_recovery_spring_only_boosts_weak_donor_before_receiver_contact(self):
        from residual_spring_support import install
        forces=torch.zeros(4,2,2,3,device='cuda')
        forces[:,0,:,0]=torch.tensor([.40,.20,.20,.20],device='cuda')[:,None]
        forces[2,1,:,0]=.1
        support=SimpleNamespace(holder=torch.tensor([1,1,1,2],device='cuda'),
            settings=dict(stiffness=300.,force_limit_n=.4,torque_limit_nm=.004))
        seen=[];support.apply=lambda:seen.append(support.settings['stiffness'].clone())
        support.reset=lambda ids:None
        support.evidence=lambda index=None:{}
        vec=SimpleNamespace(state=dict(forces=forces),progress=SimpleNamespace(flags=torch.ones(4,4,device='cuda',dtype=torch.bool)))
        install(support,900,vec,True);support.apply()
        torch.testing.assert_close(seen[0].flatten(),torch.tensor([300.,900.,300.,300.],device='cuda'))
        # Recovery remains engaged after pressure returns; receiver contact ends it.
        forces[1,0,:,0]=.4;support.apply();self.assertEqual(float(seen[-1][1]),900.)
        forces[1,1,:,0]=.1;support.apply();self.assertEqual(float(seen[-1][1]),300.)
        support.reset(torch.tensor([1],device='cuda'));self.assertEqual(float(support.spring_context[1].sum()),0.)

    def test_aperture_recovery_never_overrides_healthy_or_shared_grasp(self):
        from residual_grasp_guard import HeldJawGuard
        guard=HeldJawGuard(1,'cuda',margin=0.)
        f=torch.zeros(1,2,2,3,device='cuda');f[:,0,0,0]=.4;f[:,0,1,0]=-.4
        state=dict(forces=f);ref=SimpleNamespace(jaw=torch.full((1,2),.13,device='cuda'))
        support=SimpleNamespace(holder=torch.ones(1,device='cuda',dtype=torch.long),recovery_active=torch.zeros(1,device='cuda',dtype=torch.bool))
        action=torch.zeros(1,14,device='cuda');action[:,6]=4*.13-1
        guard.apply(action,state,ref,support,True)
        f.mul_(.1);ref.jaw[:,0]=.12;action[:,6]=4*.09-1
        self.assertTrue(torch.equal(guard.apply(action,state,ref,support,True),action))
        support.recovery_active.fill_(True)
        self.assertGreater(float(guard.apply(action,state,ref,support,True)[0,6]),float(action[0,6]))
        f[:,1,:,0]=.1
        self.assertTrue(torch.equal(guard.apply(action,state,ref,support,True),action))

    def test_pending_lift_recovery_requires_measured_grasp_height_and_pressure(self):
        from residual_spring_support import install
        n=5
        f=torch.zeros(n,2,2,3,device='cuda');f[:,0,:,0]=.20;f[3,0,:,0]=.40;f[4,1,:,0]=.10
        flags=torch.zeros(n,4,device='cuda',dtype=torch.bool);flags[:,0]=True;flags[0,0]=False
        bottom=torch.tensor([.05,.042,.044,.044,.044],device='cuda')
        support=SimpleNamespace(holder=torch.ones(n,device='cuda',dtype=torch.long),settings=dict(stiffness=300.,force_limit_n=.4,torque_limit_nm=.004))
        seen=[];support.apply=lambda:seen.append(support.settings['stiffness'].clone());support.reset=lambda ids:None;support.evidence=lambda index=None:{}
        vec=SimpleNamespace(state=dict(forces=f,bottom=bottom),progress=SimpleNamespace(flags=flags))
        install(support,900,vec,True,True);support.apply()
        torch.testing.assert_close(seen[-1].flatten(),torch.tensor([300.,300.,900.,300.,300.],device='cuda'))
        self.assertFalse(bool(flags[:,1].any()))
        self.assertTrue(support.evidence()['pressure_recovery_before_lift_confirmation'])
        # The receiver's actual contact ends the aid; no success is assigned.
        f[2,1,:,0]=.01;support.apply();self.assertEqual(float(seen[-1][2]),300.)
        self.assertFalse(bool(flags[:,1].any()))

    def test_repair_fork_keeps_weights_and_parent_sources_immutable(self):
        from residual_recovery_retry import fork_candidate
        from residual_pipeline import sha
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);parent=root/'parent';sources=root/'new_sources'
            (parent/'base').mkdir(parents=True);(parent/'sources').mkdir();sources.mkdir()
            (parent/'candidate.zip').write_bytes(b'unchanged SAC weights')
            (parent/'base/policy.pt').write_bytes(b'unchanged BC weights')
            (parent/'sources/residual_main.py').write_text('old runtime')
            (sources/'residual_main.py').write_text('new runtime')
            saved=dict(sac_sha256=sha(parent/'candidate.zip'),
                base_hashes={'policy.pt':sha(parent/'base/policy.pt')},
                source_hashes={'residual_main.py':sha(parent/'sources/residual_main.py')},
                recovery_before_lift=False,training_parameters={})
            (parent/'manifest.json').write_text(json.dumps(saved));digest=sha(parent/'manifest.json')
            candidate=fork_candidate(parent/'candidate.zip',root/'repair',sources)
            self.assertEqual(sha(candidate),saved['sac_sha256'])
            self.assertEqual(sha(parent/'manifest.json'),digest)
            fork=json.loads((candidate.parent/'manifest.json').read_text())
            self.assertTrue(fork['recovery_before_lift']);self.assertFalse(fork['additional_training'])
            self.assertNotEqual(fork['source_hashes'],saved['source_hashes'])



if __name__=='__main__':unittest.main(verbosity=2)
