"""CUDA checks of expert-free inputs and measured pickup supervision."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
from rl_core import *
from rl_data import trace_state
from autonomy_learning import physical_observation,Policy,PosePolicy,OptionPosePolicy,MemoryPolicy,load_policy,SCHEMA,future_actions,future_poses,option_entry_observations,completed_option_pose_mask
from autonomy_teacher import PickupTeacher
from autonomy_full_teacher import FullTeacher
from autonomy_env import ContactMotorReference,AutonomyProgress

ROOT=Path(__file__).resolve().parents[2]
class AutonomyChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(4);cls.device='cuda:0'
        cls.profile=json.loads((ROOT/'tools/peg_grasp_v2/peg_profile.json').read_text())
        with (ROOT/'logs/grasp_v2/full_assisted_validation3_2026-10-05.trace.jsonl').open() as f:cls.row=json.loads(next(f))
    def env(self):
        state=trace_state(self.row['pre_action_state'],None,self.device)
        progress=Progress(1,self.device,self.profile);progress.reset(torch.tensor([0],device=self.device),state)
        return SimpleNamespace(num_envs=1,device=self.device,state=state,profile=self.profile,progress=progress)
    def test_actor_inputs_exclude_expert_pose_leads(self):
        env=self.env();ref=MotorReference(env.state)
        before=physical_observation(observe(env.state,ref,env.progress))
        ref.p+=.01;ref.q=quaternion(torch.ones(1,2,3,device=self.device))
        after=physical_observation(observe(env.state,ref,env.progress))
        torch.testing.assert_close(before,after);self.assertEqual(before.shape,(1,98))
    def spring_env(self):
        class Object:
            def __init__(obj):
                obj.data=SimpleNamespace(root_pos_w=torch.zeros(2,3,device=self.device),root_quat_w=torch.tensor([[1.,0,0,0]]*2,device=self.device),root_lin_vel_w=torch.zeros(2,3,device=self.device),root_ang_vel_w=torch.zeros(2,3,device=self.device))
                obj.last=None
            def write_data_to_sim(obj):pass
            def set_external_force_and_torque(obj,force,torque,**kwargs):
                self.assertTrue(kwargs['is_global']);obj.last=(force.clone(),torque.clone())
        obj=Object();scene={'object':obj}
        for arm in (1,2):scene[f'ee_{arm}_frame']=SimpleNamespace(data=SimpleNamespace(target_pos_w=torch.zeros(2,1,3,device=self.device),target_quat_w=torch.tensor([[[1.,0,0,0]]]*2,device=self.device)))
        return SimpleNamespace(num_envs=2,device=self.device,physics_dt=.0025,scene=scene)
    def test_grasp_spring_requires_contacts_and_transfers_then_releases(self):
        from autonomy_support import GraspSpring
        env=self.spring_env();spring=GraspSpring(env,1.)
        state=dict(tool_q=torch.tensor([[[1.,0,0,0]]*2]*2,device=self.device),peg_q=env.scene['object'].data.root_quat_w,
            peg=torch.zeros(2,3,device=self.device),tip=torch.zeros(2,2,3,device=self.device),bottom=torch.ones(2,device=self.device)*.05,forces=torch.zeros(2,2,2,3,device=self.device))
        contact=torch.zeros(2,2,dtype=torch.bool,device=self.device)
        progress=SimpleNamespace(contact=lambda s:contact,contact_age=torch.ones(2,2,device=self.device),flags=torch.ones(2,4,dtype=torch.bool,device=self.device))
        reference=SimpleNamespace(jaw=torch.ones(2,2,device=self.device)*.15)
        spring.observe(state,progress,reference);self.assertFalse(bool(spring.holder.any()))
        contact[:,0]=True;state['forces'][:,0,0,0]=.4;state['forces'][:,0,1,0]=-.4
        spring.observe(state,progress,reference);torch.testing.assert_close(spring.holder,torch.ones(2,dtype=torch.long,device=self.device))
        contact[:,1]=True;state['forces'][:,1,0,0]=.4;state['forces'][:,1,1,0]=-.4
        reference.jaw[:,0]=.2;spring.observe(state,progress,reference)
        torch.testing.assert_close(spring.holder,torch.ones(2,dtype=torch.long,device=self.device)*2)
        state['forces'][:,1]=0;reference.jaw[:,1]=.2;spring.observe(state,progress,reference)
        self.assertFalse(bool(spring.holder.any()));self.assertFalse(bool(env.scene['object'].last[0].any()))
    def test_grasp_spring_bounds_force_and_reset_preserves_other_environment(self):
        from autonomy_support import GraspSpring
        env=self.spring_env();spring=GraspSpring(env,.5);spring.holder[:]=1
        obj=env.scene['object'];obj.data.root_pos_w[:,0]=-.02
        obj.data.root_quat_w[:]=quaternion(torch.ones(2,3,device=self.device))
        position=obj.data.root_pos_w.clone();orientation=obj.data.root_quat_w.clone()
        obj.write_data_to_sim()
        self.assertTrue(bool((spring.force[:,0].norm(dim=-1)<=.075001).all()))
        self.assertTrue(bool((spring.torque[:,0].norm(dim=-1)<=.001001).all()))
        self.assertTrue(bool((spring.force[:,0,0]>0).all()))
        torch.testing.assert_close(obj.data.root_pos_w,position);torch.testing.assert_close(obj.data.root_quat_w,orientation)
        other=spring.force[1].clone();spring.reset(torch.tensor([0],device=self.device))
        self.assertEqual(int(spring.holder[0]),0);self.assertEqual(int(spring.holder[1]),1)
        torch.testing.assert_close(spring.force[1],other)
        spring.close();self.assertFalse(bool(obj.last[0].any()))
    def test_zero_grasp_assistance_and_native_gate_rejects_assisted_results(self):
        from autonomy_support import GraspSpring
        from autonomy_full_recovery import native_full_gate
        env=self.spring_env();spring=GraspSpring(env,0.);spring.apply()
        self.assertIsNone(env.scene['object'].last)
        self.assertFalse(spring.evidence()['assisted_grasp'])
        self.assertFalse(native_full_gate(dict(complete=True,skill='full',assisted_grasp=True,teacher_at_inference=False),3,1.))
        with self.assertRaises(ValueError):GraspSpring(env,float('nan'))
    def test_firm_grasp_force_bounds_and_native_placement_rejects_active_guide(self):
        from autonomy_support import GraspSpring
        env=self.spring_env();spring=GraspSpring(env,1.,'firm');spring.holder[:]=1
        env.scene['object'].data.root_pos_w[:,0]=-.02;spring.apply()
        self.assertTrue(bool((spring.force[:,0].norm(dim=-1)<=.400001).all()))
        self.assertGreater(float(spring.force[0,0,0]),.15)
        p=AutonomyProgress(2,self.device,self.profile)
        target=torch.tensor([[.042,.015,.01]]*2,device=self.device)
        s=dict(peg_q=torch.tensor([[1.,0,0,0]]*2,device=self.device),peg=target-p.hole,
            target=target,bottom=torch.ones(2,device=self.device)*.01,speed=torch.zeros(2,device=self.device),
            forces=torch.zeros(2,2,2,3,device=self.device))
        self.assertTrue(bool(p.placement(s)[0].all()))
        p.grasp_support=SimpleNamespace(holder=torch.zeros(2,dtype=torch.long,device=self.device),
            placement_active=torch.tensor([True,False],device=self.device))
        torch.testing.assert_close(p.placement(s)[0],torch.tensor([False,True],device=self.device))
    def test_placement_guide_requires_free_handover_and_releases_at_center(self):
        from autonomy_support import GraspSpring
        env=self.spring_env()
        class Scene(dict):pass
        env.scene=Scene(env.scene);env.scene.env_origins=torch.zeros(2,3,device=self.device)
        env.profile=dict(hole_center_m=[0.,0.])
        spring=GraspSpring(env,.5,'firm',True)
        q=torch.tensor([[1.,0,0,0]]*2,device=self.device)
        position=torch.tensor([[.002,0,.02]]*2,device=self.device)
        env.scene['object'].data.root_pos_w[:]=position
        s=dict(tool_q=q[:,None].expand(-1,2,-1),peg_q=q,peg=position.clone(),tip=torch.zeros(2,2,3,device=self.device),
            target=torch.zeros(2,3,device=self.device),bottom=torch.ones(2,device=self.device)*.01,
            speed=torch.zeros(2,device=self.device),forces=torch.zeros(2,2,2,3,device=self.device))
        progress=SimpleNamespace(contact=lambda s:torch.zeros(2,2,dtype=torch.bool,device=self.device),
            contact_age=torch.zeros(2,2,device=self.device),flags=torch.zeros(2,4,dtype=torch.bool,device=self.device))
        reference=SimpleNamespace(jaw=torch.ones(2,2,device=self.device)*.5)
        spring.observe(s,progress,reference);self.assertFalse(bool(spring.placement_active.any()))
        progress.flags[:,3]=True;s['forces'][1,1,0,0]=.1
        spring.observe(s,progress,reference);torch.testing.assert_close(spring.placement_active,torch.tensor([True,False],device=self.device))
        spring.apply();self.assertLess(float(spring.force[0,0,0]),0)
        self.assertEqual(float(spring.force[0,0,2]),0)
        self.assertLessEqual(float(spring.force[0,0].norm()),.075001)
        torch.testing.assert_close(env.scene['object'].data.root_pos_w,position)
        s['forces'][:]=0
        with torch.inference_mode():
            spring.observe(s,progress,reference);spring.apply()
        spring.reset(torch.tensor([0],device=self.device));self.assertFalse(bool(spring.placement_active[0]));self.assertTrue(bool(spring.placement_active[1]))
        s['peg'][:,:2]=0;env.scene['object'].data.root_pos_w[:,:2]=0
        spring.observe(s,progress,reference);spring.apply();self.assertFalse(bool(spring.placement_active.any()))
        self.assertFalse(bool(spring.force.any()));self.assertGreater(int(spring.placement_steps[1]),0)
        spring.close();self.assertFalse(bool(env.scene['object'].last[0].any()))
        with self.assertRaises(ValueError):GraspSpring(env,0.,'firm',True)
    def test_assistance_curriculum_requires_actual_release_and_matching_strength(self):
        from autonomy_assistance_curriculum import assisted_gate
        row=dict(success=True,full_success=True,handover=True,assisted_grasp=True,failure=False,timeout=False,
            peak_force_n=.5,hole_error_mm=.5,terminal_physics=dict(bottom_m=.01,forces_n=[[0.,0.],[0.,0.]]),
            grasp_support=dict(strength=.5,applied_physics_steps=20,holder=0,force_peak_n=.07,torque_peak_nm=.0009,
                object_pose_writes=False,object_velocity_writes=False))
        report=dict(complete=True,skill='full',episodes=3,success=3,full_success=3,teacher_at_inference=False,
            programmed_task_controller=False,assisted_grasp=True,grasp_support=dict(strength=.5),cases=[row]*3)
        self.assertTrue(assisted_gate(report,.5,3,1.));self.assertFalse(assisted_gate(report,0.,3,1.))
        row['grasp_support']['holder']=2;self.assertFalse(assisted_gate(report,.5,3,1.))
        row['grasp_support']['holder']=0;self.assertFalse(assisted_gate(report,1.,3,1.))
    def test_constraints_gate_rejects_active_guide_and_wrong_force_preset(self):
        from autonomy_constraints_gate import assisted_gate
        support=dict(strength=1.,profile='firm',holder=0,applied_physics_steps=100,
            placement_guide_enabled=True,placement_guide_active=False,force_peak_n=.35,torque_peak_nm=.003,
            placement_force_peak_n=.12,placement_torque_peak_nm=.001,object_pose_writes=False,object_velocity_writes=False)
        row=dict(success=True,full_success=True,handover=True,assisted_grasp=True,failure=False,timeout=False,
            peak_force_n=.9,hole_error_mm=.8,terminal_physics=dict(bottom_m=.01,forces_n=[[0.,0.],[0.,0.]]),grasp_support=support)
        config=dict(strength=1.,profile='firm',placement_guide_enabled=True,force_limit_n=.4,torque_limit_nm=.004)
        r=dict(complete=True,skill='full',episodes=3,success=3,full_success=3,assisted_grasp=True,
            teacher_at_inference=False,programmed_task_controller=False,grasp_support=config,cases=[row]*3)
        self.assertTrue(assisted_gate(r,1.,3,1.))
        support['placement_guide_active']=True;self.assertFalse(assisted_gate(r,1.,3,1.))
        support['placement_guide_active']=False;config['force_limit_n']=10.;self.assertFalse(assisted_gate(r,1.,3,1.))
        config['force_limit_n']=.4;self.assertFalse(assisted_gate(r,0.,3,1.))
    def test_contact_pose_head_is_bounded_and_rotation_invariant(self):
        from autonomy_learning import PegContactPoseHead
        obs=physical_observation(observe(self.env().state,MotorReference(self.env().state),self.env().progress))
        head=PegContactPoseHead(19).to(self.device)
        option=torch.tensor([10],device=self.device);local=torch.zeros(1,7,device=self.device);local[:,3]=1
        original=head(obs,option,local)
        self.assertLessEqual(float(original.norm(dim=-1).max()),.001001)
        rotation=quaternion(torch.tensor([[.1,-.2,.4]],device=self.device))
        turned=obs.clone()
        for section in (slice(35,38),slice(70,73)):turned[:,section]=apply(rotation,obs[:,section])
        turned[:,73:77]=multiply(rotation,obs[:,73:77])
        turned[:,56:62]=apply(rotation[:,None].expand(-1,2,-1),obs[:,56:62].reshape(-1,2,3)).reshape(-1,6)
        torch.testing.assert_close(head(turned,option,local),original,atol=1e-8,rtol=1e-4)
    def test_contact_head_save_load_keeps_other_behavior_goals(self):
        from autonomy_learning import PegContactPoseHead
        obs=physical_observation(observe(self.env().state,MotorReference(self.env().state),self.env().progress))
        net=OptionPosePolicy(obs[0],torch.ones(98,device=self.device),options=19,linear_goals=True).to(self.device)
        net.relative_pose_mask[10:12,1]=True
        before=net.outputs(obs)[1].detach().clone()
        net.contact_pose_head=PegContactPoseHead(19).to(self.device);net.contact_pose_mask[10:12]=True
        after=net.outputs(obs)[1]
        torch.testing.assert_close(after[:,:10],before[:,:10]);torch.testing.assert_close(after[:,12:],before[:,12:])
        torch.testing.assert_close(after[:,10:12,:8],before[:,10:12,:8])
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            path=Path(folder)/'policy.pt';torch.save(dict(schema=SCHEMA,architecture='linear_options',state_dict=net.state_dict()),path)
            restored=load_policy(path,self.device)
            torch.testing.assert_close(restored.outputs(obs)[1],after)
    def test_contact_labels_require_whole_validation_and_complete_collection(self):
        import numpy as np
        from autonomy_learning import contact_pose_examples
        net=OptionPosePolicy(torch.zeros(98,device=self.device),torch.ones(98,device=self.device),options=19,linear_goals=True).to(self.device)
        obs=torch.zeros(8,98);obs[:,73]=1;obs[:,35]=.01
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            path=Path(folder)/'demo.npz'
            meta=dict(schema=SCHEMA,skill='full',assisted_grasp=False,collection_complete=True,episodes=[0,1])
            arrays=dict(obs=obs.numpy(),actions=torch.zeros(8,14).numpy(),stage=np.tile([10,10,11,12],2),episode=np.repeat([0,1],4),validation=np.repeat([False,True],4))
            np.savez(path,**arrays,metadata=json.dumps(meta))
            examples=contact_pose_examples([path],net,self.device)
            self.assertEqual(set(examples['episode'][~examples['validation']].tolist()),{0})
            self.assertEqual(set(examples['episode'][examples['validation']].tolist()),{1})
            arrays['validation'][1]=True;np.savez(path,**arrays,metadata=json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'Whole-episode'):contact_pose_examples([path],net,self.device)
            meta['collection_complete']=False;np.savez(path,**arrays,metadata=json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'Complete native'):contact_pose_examples([path],net,self.device)
    def test_contact_correction_does_not_follow_tool_or_object_displacement(self):
        from autonomy_learning import PegContactPoseHead
        obs=physical_observation(observe(self.env().state,MotorReference(self.env().state),self.env().progress))
        head=PegContactPoseHead(19).to(self.device);option=torch.tensor([10],device=self.device)
        pose=torch.zeros(1,7,device=self.device);pose[:,3]=1
        original=head(obs,option,pose)
        disturbed=obs.clone();disturbed[:,35:38]+=.01;disturbed[:,70:73]-=.02
        torch.testing.assert_close(head(disturbed,option,pose),original)
    def test_archive_requires_both_seeds_and_unchanged_native_reports(self):
        from preserve_autonomy import certification_evidence,sha
        case=dict(success=True,full_success=True,handover=True,failure=False,timeout=False,peak_force_n=.4,
            hole_error_mm=.5,terminal_physics=dict(bottom_m=.01,forces_n=[[0.,0.],[0.,0.]]))
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            root=Path(folder);policy=root/'policy.pt';policy.write_bytes(b'fixed weights');digest=sha(policy)
            def report(path,count,seed):
                path.write_text(json.dumps(dict(complete=True,skill='full',episodes=count,success=count,full_success=count,
                    teacher_at_inference=False,assisted_grasp=False,programmed_task_controller=False,
                    policy_sha256=digest,seed=seed,cases=[case]*count)))
            pilot=root/'pilot.json';report(pilot,3,123)
            results=[]
            for seed in (123,9876):
                path=root/f'{seed}.json';report(path,16,seed)
                results.append(dict(seed=seed,passed=True,report=str(path),report_sha256=sha(path)))
            status=dict(complete=True,native_full_policy_verified=True,stage='certified',policy_sha256=digest,
                pilot_report=str(pilot),pilot_report_sha256=sha(pilot),results=results)
            path=root/'status.json';path.write_text(json.dumps(status))
            self.assertEqual(len(certification_evidence(policy,root)[1]),3)
            status['results']=results[:1];path.write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError,'Both independent'):certification_evidence(policy,root)
            status['results']=results;path.write_text(json.dumps(status))
            changed=root/'9876.json';r=json.loads(changed.read_text());r['assisted_grasp']=True;changed.write_text(json.dumps(r))
            results[1]['report_sha256']=sha(changed);path.write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError,'Invalid or changed'):certification_evidence(policy,root)
    def test_archive_verification_detects_changed_weights(self):
        import preserve_autonomy as archive
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            root=Path(folder);bundle=root/'bundle';bundle.mkdir();policy=bundle/'policy.pt';policy.write_bytes(b'original')
            physics=root/'physics.json';physics.write_text('{}')
            path=root/'milestone.json';path.write_text(json.dumps(dict(preserved_bundle='bundle',
                physics_manifest='physics.json',physics_manifest_sha256=archive.sha(physics),files={'policy.pt':archive.sha(policy)})))
            with patch.object(archive,'ROOT',root):
                archive.verify_archive(path);policy.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'Preserved artifact'):archive.verify_archive(path)
    def test_contiguous_dagger_takeover_handles_asynchronous_reset(self):
        from autonomy_main import learner_execution_mask
        age=torch.tensor([1200,500,1300],device=self.device)
        stage=torch.tensor([3,4,2],device=self.device)
        mask=learner_execution_mask(age,stage,.5,4,1200)
        torch.testing.assert_close(mask,torch.tensor([False,True,False],device=self.device))
        age[0]=0
        torch.testing.assert_close(learner_execution_mask(age,stage,.5,4,1200),torch.tensor([True,True,False],device=self.device))
        age[0]=1200
        self.assertFalse(bool(learner_execution_mask(age,stage,.5,4,1200)[0]))
    def test_router_labels_are_causal_and_keep_interleaved_retry_context(self):
        from autonomy_learning import causal_router_examples
        data=dict(stage=torch.tensor([4,2,5,3,6,4,0,4],device=self.device),
            episode=torch.tensor([0,1,0,1,0,1,0,1],device=self.device))
        examples=causal_router_examples(data)
        changed=examples['label']!=examples['context']
        actual=data['stage'][examples['indices'][changed]]
        torch.testing.assert_close(actual,examples['label'][changed])
        for index,context,entry in zip(examples['indices'][changed].tolist(),examples['context'][changed].tolist(),examples['entry'][changed].tolist()):
            self.assertEqual(int(data['episode'][index]),int(data['episode'][entry]))
            self.assertLess(entry,index)
            self.assertEqual(int(data['stage'][entry]),context)
        self.assertTrue(bool(((examples['context']==6)&(examples['label']==0)).any()))
        self.assertFalse(bool(((examples['indices']<4)&(examples['label']==6)).any()))
    def test_router_uses_goal_context_and_has_cuda_backward_and_save_load(self):
        from autonomy_learning import LearnedOptionRouter
        torch.manual_seed(42)
        net=OptionPosePolicy(torch.zeros(98,device=self.device),torch.ones(98,device=self.device),
            options=19,linear_goals=True,option_router=True).to(self.device)
        obs=torch.zeros(2,98,device=self.device);obs[:,3]=1;obs[:,38]=1
        goal=torch.zeros(2,16,device=self.device);goal[:,3]=1;goal[:,11]=1
        option=torch.tensor([2,2],device=self.device)
        before=net.option_router(obs,option,goal,net.mean,net.std,net.input_mask)
        goal[1,0]=.02
        after=net.option_router(obs,option,goal,net.mean,net.std,net.input_mask)
        self.assertGreater(float((after[1]-before[1]).abs().sum()),0)
        translated=obs.clone();translated_goal=goal.clone()
        translated[:,0:3]+=.3;translated_goal[:,0:3]+=.3
        translated[:,35:38]+=.3;translated_goal[:,8:11]+=.3
        torch.testing.assert_close(after,net.option_router(translated,option,translated_goal,net.mean,net.std,net.input_mask),atol=1e-6,rtol=1e-5)
        for p in net.parameters():p.requires_grad_(False)
        for p in net.option_router.parameters():p.requires_grad_(True)
        loss=net.option_router(obs,option,goal,net.mean,net.std,net.input_mask).square().mean();loss.backward()
        self.assertTrue(any(p.grad is not None for p in net.option_router.parameters()))
        self.assertTrue(all(p.grad is None for p in net.pose_head.parameters()))
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            path=Path(folder)/'policy.pt';torch.save(dict(schema=SCHEMA,architecture='linear_options',state_dict=net.state_dict()),path)
            loaded=load_policy(path,self.device)
            torch.testing.assert_close(after,loaded.option_router(obs,option,goal,loaded.mean,loaded.std,loaded.input_mask))
        net.latch_option_poses.fill_(True);net.initial_option.fill_(0)
        with torch.no_grad():
            net.poses(obs);net.reset(torch.tensor([0],device=self.device));net.poses(obs)
            net.poses(obs[:1]);net.poses(obs)
    def test_router_contact_features_are_invariant_to_force_direction(self):
        from autonomy_learning import LearnedOptionRouter
        router=LearnedOptionRouter(19).to(self.device)
        obs=torch.zeros(2,98,device=self.device);obs[:,3]=1;obs[:,38]=1
        obs[:,21]=.38;obs[:,24]=-.4;obs[:,56]=.37;obs[:,59]=-.39
        goal=torch.zeros(2,16,device=self.device);goal[:,3]=1;goal[:,11]=1
        option=torch.tensor([3,11],device=self.device);mean=torch.zeros(98,device=self.device);std=torch.ones_like(mean)
        baseline=router(obs,option,goal,mean,std,std)
        rotated=obs.clone()
        for start in (21,24,56,59):
            rotated[:,start:start+3]=obs[:,start:start+3][:,[1,2,0]]
        torch.testing.assert_close(baseline,router(rotated,option,goal,mean,std,std))
        rotated[:,19:21]=.8;rotated[:,54:56]=.4;rotated[:,34]=.7;rotated[:,69]=.4
        torch.testing.assert_close(baseline,router(rotated,option,goal,mean,std,std))
    def test_router_force_calibration_preserves_measured_progress_and_negative_cases(self):
        from autonomy_learning import calibrate_router_context
        obs=torch.zeros(3,98,device=self.device);obs[:,3]=1;obs[:,38]=1
        obs[:,21]=.31;obs[:,24]=-.32;obs[:,56]=.37;obs[:,59]=-.38
        obs[:,90:98]=torch.arange(8,device=self.device)/8
        goal=torch.zeros(3,16,device=self.device)
        context=torch.tensor([3,10,3],device=self.device);label=torch.tensor([4,11,3],device=self.device)
        calibrated,targets=calibrate_router_context(obs,goal,context,label)
        torch.testing.assert_close(calibrated[:,90:98],obs[:,90:98])
        torch.testing.assert_close(calibrated[2],obs[2]);torch.testing.assert_close(targets[2],goal[2])
        for row,arm,minimum in ((0,0,.30),(1,1,.36)):
            f=calibrated[row,arm*35+21:arm*35+27].reshape(2,3)
            self.assertGreaterEqual(float(f.norm(dim=-1).min()),minimum-1e-6)
            self.assertLessEqual(float(f.norm(dim=-1).max()),minimum+.20+1e-6)
            self.assertLess(float((f[0]*f[1]).sum()),0)
    def test_full_certification_rejects_assistance_and_disturbed_placement(self):
        from autonomy_full_recovery import native_full_gate
        import copy
        row=dict(success=True,full_success=True,handover=True,failure=False,timeout=False,peak_force_n=.7,
            hole_error_mm=.8,terminal_physics=dict(bottom_m=.010,forces_n=[[0.,0.],[0.,0.]]))
        report=dict(complete=True,skill='full',teacher_at_inference=False,assisted_grasp=False,
            cases=[copy.deepcopy(row) for _ in range(16)],episodes=16,success=16,full_success=16)
        self.assertTrue(native_full_gate(report))
        report['cases'][0]['terminal_physics']['bottom_m']=.035
        self.assertFalse(native_full_gate(report))
        report['cases'][0]=copy.deepcopy(row);report['teacher_at_inference']=True
        self.assertFalse(native_full_gate(report))
        report['teacher_at_inference']=False;report['complete']=False
        self.assertFalse(native_full_gate(report))
    def test_receiver_opening_calibration_preserves_loaded_donor(self):
        from autonomy_learning import calibrate_opening_observations,full_teacher_pressure_rates
        obs=torch.zeros(3,98,device=self.device)
        obs[:,34]=.13/.5;obs[:,19:21]=.13/.5
        obs[:,21]=.39;obs[:,24]=-.39
        obs[:,69]=.25/.5;obs[:,54:56]=.25/.5
        stage=torch.tensor([0,5,10],device=self.device)
        refs=torch.tensor([[.05,.4],[.45,.1],[.2,.4]],device=self.device)
        calibrated=calibrate_opening_observations(obs,stage,refs,full=True)
        torch.testing.assert_close(calibrated[1,:35],obs[1,:35])
        torch.testing.assert_close(calibrated[2],obs[2])
        action=torch.zeros(3,14,device=self.device);action[:,13]=0
        rates=full_teacher_pressure_rates(calibrated,action,stage,.36,.40)
        self.assertLess(float(rates[0,1]),0)
        self.assertGreater(float(rates[1,1]),0)
        self.assertEqual(float(rates[1,0]),0)
    def test_teacher_starts_moving_without_a_clock(self):
        env=self.env();teacher=PickupTeacher(env)
        action=teacher.action();self.assertGreater(float(action[0,:3].norm()),0.)
        self.assertTrue(bool(torch.isfinite(action).all()));self.assertLessEqual(float(action.abs().max()),1.)
    def test_latched_pose_labels_use_each_visit_entry_in_its_own_episode(self):
        obs=torch.arange(8,device=self.device)[:,None].expand(-1,98).float()
        data=dict(obs=obs,episode=torch.tensor([0,1,0,1,0,1,0,1],device=self.device))
        stages=torch.tensor([0,0,0,1,1,1,0,0],device=self.device)
        entry=option_entry_observations(data,stages)
        torch.testing.assert_close(entry[:,0],torch.tensor([0,1,0,3,4,3,6,7],device=self.device).float())
        torch.testing.assert_close(data['obs'],obs)
    def test_retry_endpoint_is_not_a_successful_lift_destination(self):
        episode=torch.tensor([0,1,0,1,0,1,0,1],device=self.device)
        stages=torch.tensor([4,3,4,4,0,5,1,5],device=self.device)
        mask=completed_option_pose_mask(dict(episode=episode),stages)
        torch.testing.assert_close(mask,torch.tensor([False,True,False,True,True,True,True,True],device=self.device))
    def test_approach_jaw_calibration_recovers_the_demonstrated_opening(self):
        from autonomy_learning import teacher_pressure_rates
        obs=torch.zeros(3,98,device=self.device);obs[:,34]=torch.tensor([.05,.15,.20],device=self.device)/.5
        obs[:,69]=.15/.5
        action=torch.zeros(3,14,device=self.device);action[:,[6,13]]=-.4
        stage=torch.tensor([0,1,2],device=self.device)
        rate=teacher_pressure_rates(obs,action,stage,.36,.40)
        torch.testing.assert_close(rate[:,0],torch.tensor([1.,0.,-1.],device=self.device),atol=3e-5,rtol=0.)
        torch.testing.assert_close(rate[:,1],torch.zeros(3,device=self.device),atol=3e-5,rtol=0.)
    def test_full_pressure_labels_release_only_with_opposed_receiver_load(self):
        from autonomy_learning import full_teacher_pressure_rates
        obs=torch.zeros(3,98,device=self.device);action=torch.zeros(3,14,device=self.device)
        obs[:,56]=.38;obs[:,59]=torch.tensor([-.38,-.05,.38],device=self.device)
        rate=full_teacher_pressure_rates(obs,action,torch.full((3,),11,device=self.device,dtype=torch.long))
        torch.testing.assert_close(rate[:,0],torch.tensor([.02/.15,-.02/.15,-.02/.15],device=self.device))
    def test_coupled_full_jaw_actor_has_slow_release_and_loadable_cuda_weights(self):
        from autonomy_learning import JawRateHead
        net=OptionPosePolicy(torch.zeros(98,device=self.device),torch.ones(98,device=self.device),
            options=19,linear_goals=True,jaw_rate_head=True,jaw_categorical=True,jaw_coupled=True)
        self.assertEqual(net.jaw_rate_head.net[0].in_features,32)
        self.assertEqual(len(net.jaw_rate_head.rates),6)
        self.assertTrue(bool((net.jaw_rate_head.rates-.02/.15).abs().min()<1e-6))
        x=torch.randn(3,98,device=self.device)
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            path=Path(folder)/'policy.pt';torch.save(dict(schema=SCHEMA,architecture='linear_options',state_dict=net.state_dict()),path)
            restored=load_policy(path,self.device)
            option=torch.tensor([3,11,16],device=self.device)
            torch.testing.assert_close(net.jaw_rate_head(x,option),restored.jaw_rate_head(x,option))
            logits=restored.jaw_rate_head(x,option,logits=True);logits.square().mean().backward()
            self.assertTrue(any(p.grad is not None for p in restored.jaw_rate_head.parameters()))
    def test_teacher_cannot_close_to_lift_without_opposing_contact(self):
        env=self.env();teacher=PickupTeacher(env);teacher.stage[:]=3
        env.state['forces'][0,0,:,0]=.5;env.state['contact_valid'][:]=True
        for _ in range(100):teacher.update()
        self.assertEqual(int(teacher.stage[0]),3)
        env.state['forces'][0,0,1,0]=-.5
        for _ in range(45):teacher.update()
        self.assertEqual(int(teacher.stage[0]),4)
    def test_async_teacher_reset_preserves_other_episode(self):
        env=self.env();env.num_envs=2;env.state={k:v.expand(2,*v.shape[1:]).clone() for k,v in env.state.items()}
        teacher=PickupTeacher(env);teacher.stage[:]=4;teacher.jaw[:]=.05
        teacher.reset(torch.tensor([0],device=self.device))
        self.assertEqual(int(teacher.stage[1]),4);self.assertEqual(int(teacher.stage[0]),0)
        torch.testing.assert_close(teacher.jaw[1],torch.tensor([.05,.05],device=self.device))
    def test_bc_save_load_preserves_both_arm_actions(self):
        net=Policy(torch.zeros(98),torch.ones(98)).to(self.device).eval();x=torch.randn(8,98,device=self.device)
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            path=Path(folder)/'policy.pt';torch.save(dict(schema=SCHEMA,state_dict=net.state_dict()),path)
            loaded=load_policy(path,self.device);torch.testing.assert_close(net(x),loaded(x))
    def test_recurrent_history_and_async_reset_on_cuda(self):
        net=MemoryPolicy(torch.zeros(98),torch.ones(98)).to(self.device).eval()
        x=torch.randn(2,8,98,device=self.device)
        with torch.inference_mode():
            batch=net(x);net.reset()
            online=torch.stack([net(x[:,t]) for t in range(8)],1)
            torch.testing.assert_close(batch,online)
            other=net.hidden[:,1].clone();net.reset(torch.tensor([0],device=self.device))
            torch.testing.assert_close(net.hidden[:,1],other)
            self.assertEqual(float(net.hidden[:,0].abs().sum()),0.)
        # Exercise an actual recurrent CUDA backward pass, not just inference.
        net.train();loss=net(x).square().mean();loss.backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in net.parameters()))
    def test_full_teacher_uses_measured_arrival_dwell(self):
        env=self.env();env.target=torch.tensor([[.042,.015,.010]],device=self.device)
        teacher=FullTeacher(env);teacher.stage[:]=7
        teacher.goal=env.state['tip'].clone();teacher.goalq=env.state['tool_q'].clone()
        for _ in range(12):teacher.update()
        self.assertEqual(int(teacher.stage[0]),8)
        teacher.stage[:]=9;teacher.dwell.zero_();teacher.goal+=.020
        for _ in range(20):teacher.update()
        self.assertEqual(int(teacher.stage[0]),9)
    def test_vertex_receiver_preserves_its_required_opening(self):
        env=self.env();env.receiver_mode='vertex';env.target=torch.tensor([[.042,.015,.010]],device=self.device)
        teacher=FullTeacher(env)
        self.assertAlmostEqual(float(teacher.jaw[0,1]),.5,places=5)
        self.assertAlmostEqual(float(teacher.action()[0,13]),1.,places=5)
        teacher.pressure(1,torch.tensor([True],device=self.device))
        self.assertGreater(float(teacher.jaw[0,1]),.49)
    def test_fixed_donor_experiment_moves_tools_only_and_preserves_donor_pose(self):
        env=self.env();env.fixed_donor_during_contact=True;env.target=env.state['target']
        teacher=FullTeacher(env);teacher.stage[:]=10
        held=teacher.hold[:,0].clone();heldq=teacher.holdq[:,0].clone()
        env.state['peg'][:,0]+=.003
        peg=env.state['peg'].clone();pegq=env.state['peg_q'].clone()
        teacher.action()
        torch.testing.assert_close(teacher.goal[:,0],held)
        torch.testing.assert_close(teacher.goalq[:,0],heldq)
        torch.testing.assert_close(env.state['peg'],peg)
        torch.testing.assert_close(env.state['peg_q'],pegq)
    def test_loaded_receiver_transform_is_measured_after_verified_handover(self):
        env=self.env();env.target=env.state['target'];env.recapture_receiver_after_transfer=True
        teacher=FullTeacher(env);teacher.stage[:]=12
        teacher.goal=env.state['tip'].clone();teacher.goalq=env.state['tool_q'].clone()
        env.progress.flags[:,3]=True
        env.progress.receiver_alone_age=torch.tensor([2.1],device=self.device)
        env.state['forces'].zero_();env.state['forces'][0,1,0,0]=.5;env.state['forces'][0,1,1,0]=-.5
        env.state['contact_valid'][:]=True
        env.state['relative'][:,1,0]+=.001
        original={key:value.clone() for key,value in env.state.items()}
        for _ in range(12):teacher.update()
        self.assertEqual(int(teacher.stage[0]),14)
        torch.testing.assert_close(teacher.relative[:,1],env.state['relative'][:,1])
        torch.testing.assert_close(teacher.relative_q[:,1],env.state['relative_q'][:,1])
        for key,value in original.items():torch.testing.assert_close(env.state[key],value)
    def test_future_targets_never_cross_episode_resets(self):
        obs=torch.zeros(6,98,device=self.device);obs[:,3]=1;obs[:,38]=1
        obs[:,0]=torch.tensor([0,.01,.02,1.,1.01,1.02],device=self.device)
        actions=torch.zeros(6,14,device=self.device);actions[:,6]=torch.arange(6,device=self.device)/10
        data=dict(obs=obs,actions=actions,episode=torch.tensor([0,0,0,1,1,1],device=self.device))
        labels=future_actions(data,2)
        self.assertEqual(float(labels[2,0]),0.);self.assertEqual(float(labels[5,0]),0.)
        torch.testing.assert_close(labels[0,6],actions[2,6])
        torch.testing.assert_close(labels[3,6],actions[5,6])
        self.assertLessEqual(float(labels.abs().max()),1.)
    def test_learned_absolute_pose_has_generic_tracking_feedback(self):
        env=self.env();ref=MotorReference(env.state)
        obs=physical_observation(observe(env.state,ref,env.progress))
        net=PosePolicy(torch.zeros(98,device=self.device),torch.ones(98,device=self.device)).to(self.device)
        goal=torch.cat((obs[:,:7],torch.zeros(1,1,device=self.device),obs[:,35:42],torch.zeros(1,1,device=self.device)),1)
        with torch.no_grad():
            for param in net.parameters():param.zero_()
            net.net[-1].bias.copy_(goal[0])
        initial=net(obs);torch.testing.assert_close(initial[:,:6],torch.zeros_like(initial[:,:6]),atol=1e-5,rtol=0)
        displaced=obs.clone();displaced[:,0]+=.01
        self.assertLess(float(net(displaced)[0,0]),-.3)
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            path=Path(folder)/'pose.pt';torch.save(dict(schema=SCHEMA,architecture='pose',state_dict=net.state_dict()),path)
            torch.testing.assert_close(load_policy(path,self.device)(displaced),net(displaced))
    def test_pose_goals_ignore_masked_inactive_arm_features(self):
        mask=torch.ones(98,device=self.device);mask[35:70]=0
        net=PosePolicy(torch.zeros(98,device=self.device),torch.ones(98,device=self.device),mask).to(self.device)
        obs=torch.randn(3,98,device=self.device);changed=obs.clone();changed[:,35:70]+=5
        torch.testing.assert_close(net.poses(obs),net.poses(changed))
    def test_contact_caps_limit_speed_without_choosing_direction(self):
        env=self.env();state=env.state
        state['forces'][0,0,0,0]=.1
        ref=ContactMotorReference(state);start=ref.p.clone();jaw=ref.jaw.clone()
        action=torch.zeros(1,14,device=self.device);action[:,0]=1;action[:,7]=-1;action[:,6]=-1
        ref.advance(action,state)
        torch.testing.assert_close(ref.p[0,0,0]-start[0,0,0],torch.tensor(.003*DT,device=self.device),atol=1e-8,rtol=1e-3)
        torch.testing.assert_close(ref.p[0,1,0]-start[0,1,0],torch.tensor(-.025*DT,device=self.device),atol=1e-8,rtol=1e-3)
        torch.testing.assert_close(jaw[0,0]-ref.jaw[0,0],torch.tensor(.02*DT,device=self.device),atol=1e-8,rtol=1e-3)
    def test_option_actor_selects_its_own_head_and_preserves_weights(self):
        net=OptionPosePolicy(torch.zeros(98,device=self.device),torch.ones(98,device=self.device),options=6,linear_goals=True).to(self.device)
        obs=torch.randn(4,98,device=self.device)
        logits,heads=net.outputs(obs)
        torch.testing.assert_close(net.poses(obs),heads[torch.arange(4,device=self.device),logits.argmax(-1)])
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            path=Path(folder)/'policy.pt';torch.save(dict(schema=SCHEMA,architecture='linear_options',state_dict=net.state_dict()),path)
            torch.testing.assert_close(load_policy(path,self.device)(obs),net(obs))
    def test_composed_pickup_preserves_each_neural_decoder_and_roundtrips(self):
        mean=torch.zeros(98,device=self.device);std=torch.ones_like(mean)
        full=OptionPosePolicy(mean,std,options=19,linear_goals=True,jaw_rate_head=True,
            jaw_categorical=True,jaw_coupled=True).to(self.device)
        pickup=OptionPosePolicy(mean+0.1,std*2,options=6,linear_goals=True,jaw_rate_head=True,
            jaw_categorical=True).to(self.device)
        obs=torch.randn(4,98,device=self.device);obs[:,3]=1;obs[:,38]=1
        with torch.inference_mode():
            old_logits,old_heads=full.outputs(obs);_,pickup_heads=pickup.outputs(obs)
            full.pickup_decoder=pickup.requires_grad_(False)
            logits,heads=full.outputs(obs)
            torch.testing.assert_close(logits,old_logits,rtol=0,atol=0)
            torch.testing.assert_close(heads[:,:6],pickup_heads,rtol=0,atol=0)
            torch.testing.assert_close(heads[:,6:],old_heads[:,6:],rtol=0,atol=0)
            for option,decoder in ((3,pickup),(10,full)):
                full.initial_option.fill_(option);full.reset()
                goals=full.poses(obs).reshape(-1,2,8)
                rates=decoder.jaw_rate_head(obs,torch.full((4,),option,device=self.device))
                expected=4*(obs[:,[34,69]]*.5+rates*.15*.01).clamp(0,.5)-1
                torch.testing.assert_close(goals[...,7],expected)
            full.reset();full.initial_option.fill_(0);full.latch_option_poses.fill_(True)
            with tempfile.TemporaryDirectory(dir='/tmp') as folder:
                path=Path(folder)/'policy.pt'
                torch.save(dict(schema=SCHEMA,architecture='linear_options',state_dict=full.state_dict()),path)
                restored=load_policy(path,self.device)
                self.assertIsNotNone(restored.pickup_decoder)
                self.assertEqual(len(restored.pickup_decoder.jaw_rate_head.rates),5)
                self.assertEqual(len(restored.jaw_rate_head.rates),6)
                torch.testing.assert_close(restored(obs),full(obs),rtol=0,atol=0)
                retained=restored.selected_goal[1:].clone()
                restored.reset(torch.tensor([0],device=self.device));restored.poses(obs)
                torch.testing.assert_close(restored.selected_goal[1:],retained)
    def test_pickup_gate_retains_neural_decisions_and_full_router_selects_continuation(self):
        from autonomy_learning import LearnedOptionRouter
        mean=torch.zeros(98,device=self.device);std=torch.ones_like(mean)
        net=OptionPosePolicy(mean,std,options=19,linear_goals=True,jaw_rate_head=True,
            jaw_categorical=True,jaw_coupled=True,pickup_decoder=True,option_router=True).to(self.device)
        obs=mean[None].repeat(2,1);obs[:,3]=1;obs[:,38]=1
        net.use_pickup_gate.fill_(True);net.initial_option.fill_(0)
        with torch.no_grad():
            for p in net.parameters():p.zero_()
            net.option_router.net[-1].bias[7]=20
            net.pickup_decoder.gate.bias[2]=2
            net.poses(obs);self.assertEqual(net.selected_option.tolist(),[0,0])
            net.poses(obs);self.assertEqual(net.selected_option.tolist(),[2,2])
            net.pickup_decoder.gate.bias[5]=4
            net.poses(obs);self.assertEqual(net.selected_option.tolist(),[5,5])
            net.allowed_transitions[5,7]=False;net.option_router.net[-1].bias[6]=10
            net.poses(obs);self.assertEqual(net.selected_option.tolist(),[5,5])
            # A forbidden full winner does not force the second option.
            net.option_router.net[-1].bias[6]=30
            net.poses(obs);self.assertEqual(net.selected_option.tolist(),[6,6])
            net.reset(torch.tensor([0],device=self.device));net.poses(obs)
            self.assertEqual(net.selected_option.tolist(),[0,6])
            with tempfile.TemporaryDirectory(dir='/tmp') as folder:
                path=Path(folder)/'policy.pt'
                torch.save(dict(schema=SCHEMA,architecture='linear_options',state_dict=net.state_dict()),path)
                restored=load_policy(path,self.device)
                self.assertTrue(bool(restored.use_pickup_gate));net.reset()
                torch.testing.assert_close(restored(obs),net(obs))
    def test_neural_contact_adjustment_removes_entry_jump_and_bounds_geometric_response(self):
        net=OptionPosePolicy(torch.zeros(98,device=self.device),torch.ones(98,device=self.device),
            options=3,linear_goals=True).to(self.device)
        obs=torch.zeros(2,98,device=self.device);obs[:,3]=1;obs[:,38]=1
        net.latch_option_poses.fill_(True);net.preserve_stationary_goals.fill_(True)
        net.track_stationary_adjustments.fill_(True)
        net.stationary_position[1,1]=True;net.stationary_orientation[1,1]=True
        net.stationary_position[2,1]=True;net.stationary_orientation[2,1]=True
        net.adjust_stationary_position[1,1]=True
        net.adjust_stationary_orientation[1,1]=True
        with torch.no_grad():
            for parameter in net.parameters():parameter.zero_()
            for option in (0,1,2):
                net.pose_head.bias[option*14+3]=1;net.pose_head.bias[option*14+10]=1
            net.pose_head.bias[7]=.2;net.pose_head.bias[21]=5
            net.pose_head.weight[21,0]=1
            net.pose_head.weight[27,0]=1
            net.gate.bias[0]=2;first=net.poses(obs).reshape(-1,2,8).clone()
            net.gate.bias[1]=5;entry=net.poses(obs).reshape(-1,2,8)
            torch.testing.assert_close(entry[:,1,:7],first[:,1,:7])
            obs[:,70]=1
            moved=net.poses(obs).reshape(-1,2,8)
            torch.testing.assert_close((moved[:,1,0]-entry[:,1,0])*.1,torch.full((2,),.002,device=self.device))
            turn=rotvec(multiply(moved[:,1,3:7],conjugate(entry[:,1,3:7]))).norm(dim=-1)
            torch.testing.assert_close(turn,torch.full((2,),.13,device=self.device))
            # The cached anchor does not accumulate repeated corrections.
            torch.testing.assert_close(net.poses(obs).reshape(-1,2,8),moved)
            net.gate.bias[2]=8
            following=net.poses(obs).reshape(-1,2,8)
            torch.testing.assert_close(following[:,1,:7],moved[:,1,:7])
            net.reset(torch.tensor([0],device=self.device));net.poses(obs)
            self.assertEqual(net.selected_option.tolist(),[2,2])
            torch.testing.assert_close(net.current_goal[1].reshape(2,8),moved[1])
            with tempfile.TemporaryDirectory(dir='/tmp') as folder:
                path=Path(folder)/'policy.pt';net.reset()
                torch.save(dict(schema=SCHEMA,architecture='linear_options',state_dict=net.state_dict()),path)
                restored=load_policy(path,self.device)
                torch.testing.assert_close(restored(obs),net(obs))
    def test_relative_bc_decoder_rotates_and_translates_without_changing_other_arm(self):
        net=OptionPosePolicy(torch.zeros(98,device=self.device),torch.ones(98,device=self.device),
            options=2,linear_goals=True).to(self.device)
        obs=torch.zeros(2,98,device=self.device);obs[:,73]=1
        before=net.outputs(obs)[1].clone()
        with torch.no_grad():net.relative_pose_goals[0,1,:3]=torch.tensor([.01,0.,0.],device=self.device)
        net.relative_pose_mask[0,1]=True
        obs[:,70:73]=torch.tensor([1.,2.,3.],device=self.device)
        rotation=quaternion(torch.tensor([[0.,0.,torch.pi/2]],device=self.device));obs[:,73:77]=rotation
        heads=net.outputs(obs)[1].reshape(-1,2,2,8)
        torch.testing.assert_close(heads[:,0,1,:3]*.1,torch.tensor([[.1,.21,.3]]*2,device=self.device))
        torch.testing.assert_close(heads[:,0,1,3:7],rotation.expand(2,-1))
        # Only the explicitly fitted receiver decoder changes.
        net.relative_pose_mask.zero_();unchanged=net.outputs(obs)[1].reshape(-1,2,2,8)
        torch.testing.assert_close(heads[:,:,0],unchanged[:,:,0]);torch.testing.assert_close(heads[:,1,1],unchanged[:,1,1])
        net.relative_pose_mask[0,1]=True
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            path=Path(folder)/'policy.pt';torch.save(dict(schema=SCHEMA,architecture='linear_options',state_dict=net.state_dict()),path)
            torch.testing.assert_close(load_policy(path,self.device).outputs(obs)[1],net.outputs(obs)[1])
    def test_relative_pose_fitting_excludes_whole_validation_episodes(self):
        from autonomy_learning import demonstrated_relative_poses
        import numpy as np
        obs=np.zeros((8,98),np.float32);obs[:,73]=1;obs[:,38]=1
        obs[:4,35]=.1;obs[4:,35]=2.
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            path=Path(folder)/'data.npz'
            np.savez(path,obs=obs,stage=np.zeros(8,np.int64),episode=np.repeat([0,1],4),
                validation=np.repeat([False,True],4),metadata=json.dumps(dict(schema=SCHEMA,skill='full',assisted_grasp=False,collection_complete=True)))
            pose,mask,evidence=demonstrated_relative_poses([path],2,1,[0],self.device)
            torch.testing.assert_close(pose[0,1,:3],torch.tensor([.01,0.,0.],device=self.device))
            self.assertTrue(bool(mask[0,1]));self.assertEqual(int(mask.sum()),1)
            self.assertEqual(evidence[0]['training_visits'],1);self.assertEqual(evidence[0]['heldout_visits'],1)
            self.assertAlmostEqual(evidence[0]['heldout_position_rmse_mm'],190.,places=3)
    def test_destination_decoder_preserves_measured_grasp_and_learned_object_goal(self):
        net=OptionPosePolicy(torch.zeros(98,device=self.device),torch.ones(98,device=self.device),options=2,linear_goals=True).to(self.device)
        obs=torch.zeros(2,98,device=self.device);obs[:,3]=1
        obs[:,70:73]=torch.tensor([1.,2.,3.],device=self.device)
        pegq=quaternion(torch.tensor([[0.,0.,torch.pi/2]],device=self.device)).expand(2,-1)
        obs[:,73:77]=pegq;obs[:,35:38]=torch.tensor([1.1,2.,3.],device=self.device)
        localq=quaternion(torch.tensor([[0.,.2,0.]],device=self.device)).expand(2,-1)
        obs[:,38:42]=multiply(pegq,localq)
        obs[:,84:87]=torch.tensor([.4,.1,.1],device=self.device)
        with torch.no_grad():net.target_object_goals[0,:3]=torch.tensor([-.006,0.,.035],device=self.device)
        net.target_object_mask[0,1]=True
        heads=net.outputs(obs)[1].reshape(-1,2,2,8)
        tool=heads[:,0,1]
        torch.testing.assert_close(tool[:,:3]*.1,torch.tensor([[.034,0.,.045]]*2,device=self.device))
        torch.testing.assert_close(tool[:,3:7],localq)
        reconstructed=multiply(tool[:,3:7],conjugate(localq))
        torch.testing.assert_close(reconstructed,torch.tensor([[1.,0.,0.,0.]]*2,device=self.device))
        net.target_object_mask.zero_();other=net.outputs(obs)[1].reshape(-1,2,2,8)
        torch.testing.assert_close(heads[:,:,0],other[:,:,0]);torch.testing.assert_close(heads[:,1,1],other[:,1,1])
        net.target_object_mask[0,1]=True
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            path=Path(folder)/'policy.pt';torch.save(dict(schema=SCHEMA,architecture='linear_options',state_dict=net.state_dict()),path)
            torch.testing.assert_close(load_policy(path,self.device).outputs(obs)[1],net.outputs(obs)[1])
    def test_destination_fitting_excludes_validation_positions(self):
        from autonomy_learning import demonstrated_target_objects
        import numpy as np
        obs=np.zeros((8,98),np.float32);obs[:,73]=1;obs[:,84:87]=[.4,.1,.1]
        obs[:4,70:73]=[.34,.1,.45];obs[4:,70:73]=[2.,2.,2.]
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            path=Path(folder)/'data.npz'
            np.savez(path,obs=obs,stage=np.zeros(8,np.int64),episode=np.repeat([0,1],4),validation=np.repeat([False,True],4),
                metadata=json.dumps(dict(schema=SCHEMA,skill='full',assisted_grasp=False,collection_complete=True)))
            goals,mask,evidence=demonstrated_target_objects([path],2,[0],self.device)
            torch.testing.assert_close(goals[0,:3],torch.tensor([-.006,0.,.035],device=self.device))
            self.assertEqual(evidence[0]['training_visits'],1);self.assertEqual(evidence[0]['heldout_visits'],1)
            self.assertGreater(evidence[0]['heldout_position_rmse_mm'],100.)
    def test_learned_decision_hysteresis_and_async_reset(self):
        net=OptionPosePolicy(torch.zeros(98,device=self.device),torch.ones(98,device=self.device),options=6).to(self.device)
        obs=torch.zeros(2,98,device=self.device)
        with torch.no_grad():
            for param in net.parameters():param.zero_()
            net.gate.bias[0]=1;net.poses(obs)
            net.gate.bias[1]=1.5;net.poses(obs)
            self.assertEqual(net.selected_option.tolist(),[0,0])
            net.gate.bias[1]=2;net.poses(obs)
            self.assertEqual(net.selected_option.tolist(),[1,1])
            net.reset(torch.tensor([0],device=self.device))
            net.gate.bias[1]=0;net.gate.bias[0]=.5;net.poses(obs)
            self.assertEqual(net.selected_option.tolist(),[0,1])
    def test_demonstrated_transition_memory_excludes_validation(self):
        from autonomy_learning import demonstrated_transitions
        import numpy as np
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            path=Path(folder)/'data.npz'
            np.savez(path,stage=np.array([0,1,2,0,2]),episode=np.array([0,0,0,1,1]),
                validation=np.array([False,False,False,True,True]),metadata=json.dumps(dict(schema=SCHEMA)))
            allowed,initial,counts=demonstrated_transitions([path],3,self.device)
            self.assertEqual(initial,0);self.assertTrue(bool(allowed[0,1]));self.assertFalse(bool(allowed[0,2]))
            net=OptionPosePolicy(torch.zeros(98,device=self.device),torch.ones(98,device=self.device),options=3).to(self.device)
            net.allowed_transitions.copy_(allowed);net.initial_option.fill_(initial)
            with torch.no_grad():
                for param in net.parameters():param.zero_()
                obs=torch.zeros(1,98,device=self.device)
                net.gate.bias[2]=10;net.poses(obs);self.assertEqual(int(net.selected_option[0]),0)
                # A forbidden winning option must not turn its second choice
                # into a forced progression through the remembered graph.
                net.gate.bias[1]=5;net.poses(obs);self.assertEqual(int(net.selected_option[0]),0)
                net.gate.bias[1]=15;net.poses(obs);self.assertEqual(int(net.selected_option[0]),1)
                net.gate.bias[2]=20
                net.poses(obs);self.assertEqual(int(net.selected_option[0]),2)
                net.gate.bias[0]=30;net.poses(obs);self.assertEqual(int(net.selected_option[0]),2)
    def test_stationary_goal_memory_preserves_contact_pose_on_option_switch_and_reset(self):
        net=OptionPosePolicy(torch.zeros(98,device=self.device),torch.ones(98,device=self.device),options=3).to(self.device)
        net.latch_option_poses.fill_(True);net.preserve_stationary_goals.fill_(True)
        net.stationary_position[1,0]=True;net.stationary_orientation[1,0]=True
        obs=torch.zeros(2,98,device=self.device);obs[:,3]=1;obs[:,38]=1
        with torch.no_grad():
            for parameter in net.parameters():parameter.zero_()
            net.gate.bias[0]=2;first=net.poses(obs).clone()
            net.gate.bias[1]=5;net.heads.bias[16:19]=.7;net.heads.bias[19]=1
            net.heads.bias[24:27]=.3
            second=net.poses(obs)
            torch.testing.assert_close(second[:,:7],first[:,:7])
            torch.testing.assert_close(second[:,8:11],torch.full((2,3),.3,device=self.device))
            net.reset(torch.tensor([0],device=self.device));obs[0,:3]=.4
            third=net.poses(obs)
            torch.testing.assert_close(third[0,:3],obs[0,:3]);torch.testing.assert_close(third[1,:7],second[1,:7])
    def test_stationary_labels_exclude_validation_and_reconstruct_motor_reference(self):
        from autonomy_learning import demonstrated_stationary_goals
        import numpy as np
        obs=np.zeros((6,98),np.float32);obs[:,3]=1;obs[:,38]=1
        obs[:,0]=[0,.001,.002,0,.1,.2]
        action=np.zeros((6,14),np.float32);action[:3,0]=-obs[:3,0]*.1/.003
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            path=Path(folder)/'demo.npz';np.savez(path,obs=obs,actions=action,stage=np.zeros(6,np.int64),
                episode=np.array([0,0,0,1,1,1]),validation=np.array([False]*3+[True]*3),
                metadata=json.dumps(dict(schema=SCHEMA,assisted_grasp=False)))
            position,orientation,evidence=demonstrated_stationary_goals([path],1,self.device)
            self.assertTrue(bool(position.all()));self.assertTrue(bool(orientation.all()))
            self.assertEqual(evidence[0]['training_visits'],1)
    def test_option_pose_memory_preserves_learned_jaw_feedback(self):
        net=OptionPosePolicy(torch.zeros(98,device=self.device),torch.ones(98,device=self.device),options=3).to(self.device)
        net.latch_option_poses.fill_(True)
        obs=torch.zeros(2,98,device=self.device)
        with torch.no_grad():
            for param in net.parameters():param.zero_()
            net.gate.bias[0]=2
            first=net.poses(obs).clone()
            net.heads.bias[0]=1;net.heads.bias[7]=.5
            second=net.poses(obs)
            torch.testing.assert_close(second[:,0],first[:,0]);self.assertEqual(float(second[0,7]),.5)
            net.reset(torch.tensor([0],device=self.device));third=net.poses(obs)
            self.assertEqual(float(third[0,0]),1);self.assertEqual(float(third[1,0]),0)
    def test_recovery_file_contains_only_completed_successful_episodes(self):
        from autonomy_main import save_completed_demonstrations
        import numpy as np
        chunks=[dict(obs=torch.zeros(6,98,device=self.device),actions=torch.zeros(6,14,device=self.device),
            stage=torch.zeros(6,dtype=torch.long,device=self.device),episode=torch.tensor([0,1,2,0,1,2],device=self.device))]
        completed={0:dict(success=True),1:dict(success=False),2:dict(success=True)}
        args=SimpleNamespace(skill='lift',learner_fraction=.1)
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            out=Path(folder)
            metadata=save_completed_demonstrations(out,chunks,completed,{0:{},2:{}},
                dict(parameters={},source_hashes={}),SCHEMA,args,partial=True)
            self.assertFalse(metadata['collection_complete'])
            with np.load(out/'completed_demonstrations.npz',allow_pickle=False) as data:
                self.assertEqual(set(data['episode'].tolist()),{0,2})
                self.assertEqual(len(data['obs']),4)
                self.assertEqual(set(data['episode'][data['validation']].tolist()),{2})
            self.assertFalse((out/'completed_demonstrations.npz.tmp').exists())
    def test_fast_success_cannot_replace_slow_allocated_failure(self):
        from autonomy_main import record_allocated_result
        completed={}
        self.assertTrue(record_allocated_result(completed,0,dict(success=True),2))
        self.assertFalse(record_allocated_result(completed,2,dict(success=True),2))
        self.assertEqual(len(completed),1)
        self.assertTrue(record_allocated_result(completed,1,dict(success=False),2))
        self.assertEqual(sum(row['success'] for row in completed.values()),1)
        with self.assertRaises(ValueError):record_allocated_result(completed,1,dict(success=True),2)
    def test_endpoint_targets_stop_at_each_contiguous_stage(self):
        obs=torch.zeros(6,98,device=self.device);obs[:,0]=torch.arange(6,device=self.device)
        data=dict(obs=obs,actions=torch.zeros(6,14,device=self.device),episode=torch.zeros(6,device=self.device),
            stage=torch.tensor([0,0,1,1,0,0],device=self.device))
        target=future_poses(data,0,True)
        torch.testing.assert_close(target[:,0],torch.tensor([1.,1.,3.,3.,5.,5.],device=self.device))
    def test_learned_jaw_rate_preserves_movement_weights_and_roundtrips(self):
        net=OptionPosePolicy(torch.zeros(98,device=self.device),torch.ones(98,device=self.device),
            options=6,linear_goals=True,jaw_rate_head=True).to(self.device)
        obs=torch.randn(4,98,device=self.device);obs[:,[34,69]]=.4
        options=torch.tensor([0,1,3,5],device=self.device)
        original={name:param.detach().clone() for name,param in net.named_parameters() if not name.startswith('jaw_rate_head.')}
        for name,param in net.named_parameters():param.requires_grad_(name.startswith('jaw_rate_head.'))
        optimizer=torch.optim.Adam(net.jaw_rate_head.parameters(),lr=3e-4)
        loss=(net.jaw_rate_head(obs,options)-.1).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        for name,param in net.named_parameters():
            if name in original:torch.testing.assert_close(param,original[name],rtol=0,atol=0)
        with torch.no_grad():
            poses=net.poses(obs).reshape(4,2,8)
            rates=net.jaw_rate_head(obs,net.selected_option)
            expected=torch.stack((obs[:,34],obs[:,69]),1)*.5+rates*.15*.01
            torch.testing.assert_close((poses[...,7]+1)*.25,expected,atol=1e-7,rtol=1e-5)
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            path=Path(folder)/'policy.pt'
            torch.save(dict(schema=SCHEMA,architecture='linear_options',state_dict=net.state_dict()),path)
            reloaded=load_policy(path,self.device)
            self.assertIsNotNone(reloaded.jaw_rate_head)
            torch.testing.assert_close(reloaded(obs),net(obs))
    def test_jaw_rate_head_depends_on_measured_contact_and_reference(self):
        from autonomy_learning import JawRateHead
        net=JawRateHead(6).to(self.device)
        obs=torch.zeros(2,98,device=self.device,requires_grad=True)
        option=torch.tensor([3,4],device=self.device)
        result=net(obs,option);result.sum().backward()
        self.assertGreater(float(obs.grad[:,[34,69]].abs().sum()),0.)
        changed=obs.detach().clone();changed[:,21]=.35;changed[:,24]=-.35
        self.assertGreater(float((net(changed,option)-result.detach()).abs().sum()),0.)
    def test_pressure_labels_and_categorical_hold_roundtrip(self):
        from autonomy_learning import teacher_pressure_rates
        obs=torch.zeros(3,98,device=self.device)
        obs[:,21]=torch.tensor([.25,.35,.50],device=self.device);obs[:,24]=-obs[:,21]
        labels=teacher_pressure_rates(obs,torch.zeros(3,14,device=self.device),torch.full((3,),3,device=self.device))
        torch.testing.assert_close(labels[:,0],torch.tensor([-.02/.15,0.,.05/.15],device=self.device))
        net=OptionPosePolicy(torch.zeros(98,device=self.device),torch.ones(98,device=self.device),
            options=6,linear_goals=True,jaw_rate_head=True,jaw_categorical=True).to(self.device)
        with torch.no_grad():
            for param in net.jaw_rate_head.parameters():param.zero_()
            net.jaw_rate_head.net[-1].bias[2]=2
            rate=net.jaw_rate_head(obs,torch.zeros(3,dtype=torch.long,device=self.device))
            torch.testing.assert_close(rate,torch.zeros_like(rate))
        logits=net.jaw_rate_head(obs,torch.zeros(3,dtype=torch.long,device=self.device),logits=True)
        torch.nn.functional.cross_entropy(logits.reshape(-1,5),torch.full((6,),2,device=self.device)).backward()
        self.assertIsNotNone(net.jaw_rate_head.net[-1].weight.grad)
        with tempfile.TemporaryDirectory(dir='/tmp') as folder:
            path=Path(folder)/'policy.pt'
            torch.save(dict(schema=SCHEMA,architecture='linear_options',state_dict=net.state_dict()),path)
            restored=load_policy(path,self.device)
            self.assertTrue(restored.jaw_rate_head.categorical)
            torch.testing.assert_close(restored(obs),net(obs))
    def test_handover_requires_receiver_alone_hold_and_resets_after_slip(self):
        env=self.env();s=env.state
        progress=AutonomyProgress(1,self.device,self.profile)
        progress.reset(torch.tensor([0],device=self.device),s)
        s['forces'].zero_();s['forces'][0,1,0,0]=.35;s['forces'][0,1,1,0]=-.35
        s['contact_valid'][:]=True;s['bottom'][:]=.045;progress.flags[:,1]=True
        action=torch.zeros(1,14,device=self.device)
        for _ in range(100):_,success,_,_=progress.update(s,action,'handover')
        self.assertFalse(bool(success[0]));self.assertFalse(bool(progress.flags[0,3]))
        s['forces'][:,1]=0;s['bottom'][:]=.010
        progress.update(s,action,'handover')
        self.assertEqual(float(progress.receiver_alone_age[0]),0.)
        s['forces'][0,1,0,0]=.35;s['forces'][0,1,1,0]=-.35;s['bottom'][:]=.045
        for _ in range(260):_,success,_,_=progress.update(s,action,'handover')
        self.assertTrue(bool(success[0]))
        progress.reset(torch.tensor([0],device=self.device),s)
        self.assertFalse(bool(progress.flags[0,3]));self.assertEqual(float(progress.receiver_alone_age[0]),0.)
    def test_teacher_preserves_donor_support_when_receiver_weakens(self):
        env=self.env();env.target=env.state['target'];teacher=FullTeacher(env);teacher.stage[:]=11
        teacher.jaw[:,0]=.15;teacher.release_start[:,0]=.13
        env.state['forces'].zero_();env.state['forces'][0,1,0,0]=.20;env.state['forces'][0,1,1,0]=-.20
        env.state['contact_valid'][:]=True
        teacher.action()
        self.assertLess(float(teacher.jaw[0,0]),.15)
        self.assertGreaterEqual(float(teacher.jaw[0,0]),.13)
    def test_board_catching_a_dropped_peg_does_not_count_as_transfer(self):
        env=self.env();s=env.state
        progress=AutonomyProgress(1,self.device,self.profile)
        progress.reset(torch.tensor([0],device=self.device),s)
        progress.flags[:,1]=True;s['forces'].zero_();s['bottom'][:]=.010
        _,success,failed,details=progress.update(s,torch.zeros(1,14,device=self.device),'full')
        self.assertTrue(bool(failed[0]));self.assertFalse(bool(success[0]))
        self.assertTrue(bool(details['dropped_before_handover'][0]))
        # After a verified transfer, a free object at board height can be
        # a valid deposit; this drop check must not prohibit placement.
        progress.flags[:,3]=True
        _,_,failed,details=progress.update(s,torch.zeros(1,14,device=self.device),'full')
        self.assertFalse(bool(details['dropped_before_handover'][0]))
    def test_curriculum_cannot_hide_a_bad_source_in_the_overall_rate(self):
        from autonomy_curriculum import pickup_gate
        cases=[dict(success=True,reset=dict(source='L5')) for _ in range(24)]
        cases += [dict(success=i<7,reset=dict(source='L6')) for i in range(8)]
        report=dict(complete=True,episodes=32,success=31,cases=cases,teacher_at_inference=False,assisted_grasp=False)
        self.assertFalse(pickup_gate(report,['L5','L6']))
        cases[-1]['success']=True;report['success']=32
        self.assertTrue(pickup_gate(report,['L5','L6']))
        report['complete']=False
        self.assertFalse(pickup_gate(report,['L5','L6']))
    def test_past_handover_cannot_authorize_retreat_after_a_slip(self):
        env=self.env();env.target=env.state['target']
        env.progress=AutonomyProgress(1,self.device,self.profile)
        env.progress.reset(torch.tensor([0],device=self.device),env.state)
        env.progress.flags[:,3]=True
        env.progress.receiver_alone_age[:]=0
        env.state['forces'].zero_();env.state['jaws'][:,0]=.3
        teacher=FullTeacher(env);teacher.stage[:]=11
        teacher.goal=env.state['tip'].clone();teacher.goalq=env.state['tool_q'].clone()
        for _ in range(12):teacher.update()
        self.assertEqual(int(teacher.stage[0]),11)
        env.state['forces'][0,1,0,0]=.5;env.state['forces'][0,1,1,0]=-.5
        env.state['contact_valid'][:]=True;env.progress.receiver_alone_age[:]=2.1
        teacher.update()
        self.assertEqual(int(teacher.stage[0]),12)
    def test_large_pose_command_preserves_its_direction(self):
        env=self.env();ref=MotorReference(env.state)
        obs=physical_observation(observe(env.state,ref,env.progress))
        net=PosePolicy(torch.zeros(98,device=self.device),torch.ones(98,device=self.device)).to(self.device)
        goal=torch.cat((obs[:,:7],torch.zeros(1,1,device=self.device),obs[:,35:42],torch.zeros(1,1,device=self.device)),1)
        goal[0,:3]+=torch.tensor([.6,0.,-.2],device=self.device)
        with torch.no_grad():
            for param in net.parameters():param.zero_()
            net.net[-1].bias.copy_(goal[0])
        action=net(obs)[0,:3]
        expected=bounded(torch.tensor([.6,0.,-.2],device=self.device),1.)
        torch.testing.assert_close(action/action.norm(),expected/expected.norm(),atol=1e-6,rtol=1e-6)

if __name__=='__main__':unittest.main(verbosity=2)
