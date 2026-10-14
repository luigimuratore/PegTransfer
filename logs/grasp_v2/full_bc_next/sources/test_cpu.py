"""Geometry and grasp-evidence checks; never starts Isaac Sim."""
import ast
import copy
import hashlib
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import sys
import importlib.util
from types import SimpleNamespace

import numpy as np
from scipy.spatial import ConvexHull
from geometry import load_profile, sectors, candidate, contact_candidate, physical_hold, valid_contact_pair, surface_label, verify_asset, preflight, closure_contact_screen
from probe import stage_limits,hold_steps,configure_camera
from motion import PoseTrajectory, MotionWatchdog, pose_reached, rotation
from scipy.spatial.transform import Rotation
from kinematics import KinematicModel, route_check
from grip import GripController, VertexGripController, grip_controller
from learning import (SCHEMA,STAGES,OBS_DIM,observation,teacher_action,motor_command,
                      save_demonstrations,load_demonstrations,MotorPolicy,goal_action,
                      LEGACY_SCHEMA,SIGNATURE_FILES,signature,dataset_experiment,feedback_action)
import json


class WallLearningTests(unittest.TestCase):
    def row(self):
        return dict(tool_tip_w_m=[.01,0,.04],tool_quaternion_wxyz=[1,0,0,0],
                    peg_in_tool_m=[0,0,-.004],peg_quat_in_tool=[1,0,0,0],
                    finger_force_vectors_n=[[.32,0,0],[-.33,0,0]],jaw_rad=[.14,.14],
                    peg_bottom_m=.05,peg_speed_m_s=0.)

    def samples(self,episodes=3):
        rng=np.random.default_rng(13);samples=[]
        for episode in range(episodes):
            for stage in STAGES:
                for _ in range(3):
                    obs=rng.normal(size=OBS_DIM).astype(np.float32)
                    obs[27:37]=np.eye(len(STAGES))[STAGES.index(stage)]
                    samples.append((episode,stage,obs,np.tanh(obs[:4]).astype(np.float32)))
        cases=[dict(episode=i,complete=True,physical_grasp_hold_success=True) for i in range(episodes)]
        return samples,cases,dict(schema=SCHEMA,mode='wall')

    def test_actor_observes_measured_state_not_clock_reward_or_success_labels(self):
        row=self.row();goal=[.02,0,.04];ref=[.0101,0,.04]
        initial=observation(row,'close',goal,[1,0,0,0],ref,.14)
        row.update(stage_elapsed_s=99.,reward=100.,physical_grasp_hold_success=True)
        np.testing.assert_array_equal(initial,observation(row,'close',goal,[1,0,0,0],ref,.14))
        self.assertEqual(initial.shape,(OBS_DIM,))
        row['finger_force_vectors_n'][0][0]=0.
        self.assertFalse(np.array_equal(initial,observation(row,'close',goal,[1,0,0,0],ref,.14)))

    def test_expert_label_and_motor_adapter_preserve_reference_speed_and_jaw(self):
        before=np.array([.01,0,.04]);after=before+[.00005,0,0]
        action=teacher_action(before,after,.15,.1498,.01,.005)
        pos,jaw=motor_command(action,before,before,.15,.01,.005,contact=True)
        np.testing.assert_allclose(pos,after,atol=1e-10)
        self.assertAlmostEqual(jaw,.1498,places=8)
        paused=teacher_action(before,after,.15,.1498,.01,.005,allow_motion=False)
        np.testing.assert_array_equal(paused[:3],np.zeros(3))
        pos,jaw=motor_command([1,1,1,-1],before,before,.15,.01,.005,True,False)
        np.testing.assert_array_equal(pos,before);self.assertLess(jaw,.15)
        ref=before.copy()
        for _ in range(1000):ref,_=motor_command([1,0,0,1],ref,before,.5,.01,.005,True)
        self.assertLessEqual(np.linalg.norm(ref-before),.0005+1e-12)
        with self.assertRaises(ValueError):motor_command([float('nan')]*4,before,before,.15,.01,.005)

    def test_positive_feedback_advances_at_the_recorded_bc1_stall(self):
        row=self.row()
        row['tool_tip_w_m']=[-.01712084,-.00028919,.05635846]
        target=np.array([-.010581685,-.000000241,.055])
        obs=observation(row,'above',target,[1,0,0,0],row['tool_tip_w_m'],.15)
        error=target-row['tool_tip_w_m']
        for latent in ([-1,-1,-1,0],[0,0,0,0],[1,1,1,0]):
            action=goal_action(obs,latent)
            self.assertGreater(np.dot(error,action[:3]),0.)
            self.assertGreater(np.linalg.norm(action[:3]),.9)
            self.assertLessEqual(np.linalg.norm(action[:3]),1.000001)
        exact=observation(row,'above',row['tool_tip_w_m'],[1,0,0,0],row['tool_tip_w_m'],.15)
        np.testing.assert_array_equal(goal_action(exact,[1,-1,1,.4])[:3],np.zeros(3))

    def test_closed_loop_arrival_with_servo_lag_and_minimum_learned_gain(self):
        # A lagging toy servo reproduces the reported start/goal separation.
        # This verifies command convergence, not contact physics in Isaac.
        row=self.row();actual=np.array([-.01712084,-.00028919,.05635846])
        target=np.array([-.010581685,-.000000241,.055]);ref=actual.copy()
        for _ in range(250):
            row['tool_tip_w_m']=actual.tolist()
            obs=observation(row,'above',target,[1,0,0,0],ref,.15)
            action=goal_action(obs,[-1,-1,-1,0])
            ref,_=motor_command(action,ref,actual,.15,.01,.025)
            actual+=.15*(ref-actual)
        self.assertLess(np.linalg.norm(target-actual),.0007)
        self.assertLess(np.linalg.norm(ref-actual),.003)

    def test_contact_feedback_obeys_speed_and_pressure_pause(self):
        row=self.row();ref=np.array(row['tool_tip_w_m']);goal=ref+[0,0,.035]
        obs=observation(row,'lift',goal,[1,0,0,0],ref,.14)
        action=goal_action(obs,[1,1,1,-.2])
        position,jaw=motor_command(action,ref,ref,.14,.01,.003,True)
        self.assertAlmostEqual(np.linalg.norm(position-ref),.00003)
        paused,_=motor_command(action,ref,ref,.14,.01,.003,True,False)
        np.testing.assert_array_equal(paused,ref)

    def test_legacy_data_upgrade_requires_intact_matching_physics_snapshot(self):
        profile=load_profile();plan=contact_candidate(profile,'wall',2,.5,1)
        params=dict(mass_g=5.,friction=.9,contact_stiffness=1000.,contact_damping=.5,
                    jaw_torque=.015,jaw_stiffness=.6,grip_force=.35,preload_seconds=.4,
                    max_force=1.,free_speed=.025,contact_speed=.005,lift_speed=.003,rotation_speed=1.)
        current=signature(SimpleNamespace(mode='wall',arm=1,**params),profile,plan)
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);snapshot=folder/'sources';snapshot.mkdir()
            for name in SIGNATURE_FILES:
                (snapshot/name).write_bytes(Path(__file__).with_name(name).read_bytes())
            with (snapshot/'learning.py').open('a') as file:file.write('\n# Older actor parameterization\n')
            prior=copy.deepcopy(current);prior['schema']=LEGACY_SCHEMA
            def checkpoint():
                prior['code_sha256']=hashlib.sha256(b''.join((snapshot/name).read_bytes() for name in SIGNATURE_FILES)).hexdigest()
                arrays=dict(metadata=json.dumps(dict(schema=LEGACY_SCHEMA,experiment=prior)),
                            mean=np.zeros(OBS_DIM),std=np.ones(OBS_DIM))
                for i,(n,m) in enumerate(((128,OBS_DIM),(128,128),(4,128))):
                    arrays[f'w{i}']=np.zeros((n,m));arrays[f'b{i}']=np.zeros(n)
                np.savez(folder/'policy.npz',**arrays)
                return dict(schema=LEGACY_SCHEMA,experiment=copy.deepcopy(prior))
            meta=checkpoint()
            with self.assertRaisesRegex(ValueError,'--initialize_from'):dataset_experiment(meta)
            self.assertEqual(dataset_experiment(meta,folder/'policy.npz'),current)
            with (snapshot/'motion.py').open('a') as file:file.write('\n# Different physical controller\n')
            with self.assertRaisesRegex(ValueError,'original experiment hash'):
                dataset_experiment(meta,folder/'policy.npz')
            meta=checkpoint()
            with self.assertRaisesRegex(ValueError,'Physical source changed: motion.py'):
                dataset_experiment(meta,folder/'policy.npz')

    def test_recorded_phase_must_match_the_observation(self):
        samples,cases,experiment=self.samples()
        samples[0][2][27:37]=np.eye(len(STAGES))[1]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.npz';save_demonstrations(path,samples,cases,experiment)
            with self.assertRaisesRegex(ValueError,'Observation phase differs'):load_demonstrations(path)

    def test_collection_rejects_failed_grasps_and_incomplete_episodes(self):
        samples,cases,experiment=self.samples()
        cases[1]['complete']=False
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.npz'
            result=save_demonstrations(path,samples,cases,experiment)
            data,meta=load_demonstrations(path)
            self.assertEqual(meta['episodes'],[0,2]);self.assertEqual(result['transitions'],60)
            self.assertNotIn(1,data['episode'])
            with self.assertRaises(FileExistsError):save_demonstrations(path,samples,cases,experiment)
            cases[0]['complete']=False;cases[2]['complete']=False
            with self.assertRaisesRegex(ValueError,'No complete physically successful'):
                save_demonstrations(Path(directory)/'failed.npz',samples,cases,experiment)
            broken,valid_cases,experiment=self.samples()
            broken=[row for row in broken if row[1]!='hold']
            partial=Path(directory)/'partial.npz'
            save_demonstrations(partial,broken,valid_cases,experiment)
            with self.assertRaisesRegex(ValueError,'Incomplete grasp sequence'):load_demonstrations(partial)

    def test_real_bc_training_export_and_numpy_inference_on_wall_dimensions(self):
        from train import train
        samples,cases,experiment=self.samples()
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            path=Path(directory)/'data.npz';output=Path(directory)/'model'
            save_demonstrations(path,samples,cases,experiment)
            manifest=train(path,output,updates=12,batch_size=16,device='cpu')
            self.assertEqual(manifest['training_episodes'],[0,1]);self.assertEqual(manifest['validation_episode'],2)
            self.assertTrue(manifest['physical_evaluation_required'])
            self.assertFalse(manifest['neural_physical_success_verified'])
            policy=MotorPolicy(output/'policy.npz',experiment)
            data,_=load_demonstrations(path);held=data['episode']==2
            errors=np.stack([policy.predict(obs) for obs in data['observations'][held]])-data['actions'][held]
            self.assertAlmostEqual(float(np.mean(errors**2)),manifest['validation']['normalized_action_mse'],places=6)
            self.assertEqual(policy.predict(data['observations'][0]).shape,(4,))
            with self.assertRaisesRegex(ValueError,'differs'):MotorPolicy(output/'policy.npz',{'mode':'vertex'})
            with self.assertRaisesRegex(ValueError,'already exists'):train(path,output,updates=1)
            samples,cases,experiment=self.samples(2);small=Path(directory)/'small.npz'
            save_demonstrations(small,samples,cases,experiment)
            with self.assertRaisesRegex(ValueError,'at least 3'):train(small,Path(directory)/'too_small',updates=1)

    def test_recorded_weak_contact_cannot_stop_closing_with_any_learned_gain(self):
        row=self.row()
        row['finger_force_vectors_n']=[[-.2983438,0.,-.0001579],[.2859401,0.,.1543264]]
        obs=observation(row,'close',row['tool_tip_w_m'],[1,0,0,0],row['tool_tip_w_m'],.1340115)
        for gain in (-1.,0.,1.):
            action=feedback_action(obs,[0,0,0,gain])
            self.assertLessEqual(float(action[3]),-.02/.15*.5+1e-7)
            pos,jaw=motor_command(action,np.array(row['tool_tip_w_m']),row['tool_tip_w_m'],.1340115,.01,.005,True)
            self.assertLess(jaw,.1340115)
            np.testing.assert_array_equal(pos,row['tool_tip_w_m'])
        row['finger_force_vectors_n']=[[.45,0,0],[-.44,0,0]]
        obs=observation(row,'close',row['tool_tip_w_m'],[1,0,0,0],row['tool_tip_w_m'],.1340115)
        self.assertGreater(float(feedback_action(obs,[0,0,0,-1])[3]),0.)
        row['finger_force_vectors_n']=[[.35,0,0],[-.35,0,0]]
        obs=observation(row,'close',row['tool_tip_w_m'],[1,0,0,0],row['tool_tip_w_m'],.1340115)
        self.assertEqual(float(feedback_action(obs,[0,0,0,1])[3]),0.)

    def test_torch_and_numpy_feedback_agree_on_pressure_boundaries(self):
        import torch
        from torch_backend import decode_action
        from learning import stage_speeds
        samples=[]
        for stage in STAGES:
            for force in (0.,.298,.35,.45):
                row=self.row();row['finger_force_vectors_n']=[[force,0,0],[-force,0,0]]
                samples.append(observation(row,stage,[.015,0,.04],[1,0,0,0],[.011,0,.04],.134))
        obs=np.stack(samples);latent=np.random.default_rng(12).uniform(-1,1,(len(obs),4)).astype(np.float32)
        expected=feedback_action(obs,latent)
        actual=decode_action(torch.from_numpy(obs),torch.from_numpy(latent),torch.from_numpy(stage_speeds())).numpy()
        np.testing.assert_allclose(actual,expected,atol=2e-7,rtol=1e-6)

    def test_pressure_actor_builds_the_missing_preload_with_a_lagging_contact(self):
        # Contact spring/servo model, not a verification of real friction.
        for latent_gain in (-1.,0.,1.):
            row=self.row();row.update(finger_contact_points_w_m=[[0,0,0],[.003,0,0]],
                                     finger_contact_surfaces=['inner','outer'])
            command=.1340115;actual=.139;guard=GripController('wall',.15);ready=False;peak=0.
            for _ in range(600):
                force=max(0.,.298+50*(.139-actual));peak=max(peak,force+.027)
                row['finger_normal_force_n']=[force,force+.027]
                row['finger_force_vectors_n']=[[force,0,0],[-force-.027,0,0]]
                obs=observation(row,'close',row['tool_tip_w_m'],[1,0,0,0],row['tool_tip_w_m'],command)
                guard.angle=command;status=guard.update(row,'close',.01)
                self.assertIsNone(status['reason'])
                if status['pressure_ready']:
                    ready=True;break
                action=feedback_action(obs,[0,0,0,latent_gain])
                _,command=motor_command(action,np.array(row['tool_tip_w_m']),row['tool_tip_w_m'],command,.01,.005,True)
                actual+=.15*(command-actual)
            self.assertTrue(ready);self.assertLess(peak,1.)

    def test_front_camera_updates_persistent_viewport_without_touching_physics(self):
        viewer=SimpleNamespace(eye=None,lookat=None)
        controller=mock.Mock();env=SimpleNamespace(cfg=SimpleNamespace(viewer=viewer),viewport_camera_controller=controller)
        eye=(.025,-.32,.25);lookat=(.025,0,.035)
        configure_camera(env,eye,lookat)
        self.assertEqual(viewer.eye,eye);self.assertEqual(viewer.lookat,lookat)
        controller.update_view_location.assert_called_once_with(eye=eye,lookat=lookat)
        env.viewport_camera_controller=None;configure_camera(env,eye,lookat)


class VertexCenteringTests(unittest.TestCase):
    def row(self,a=0.,b=0.):
        return dict(finger_normal_force_n=[a,b],finger_force_vectors_n=[[-a,0,0],[b,0,0]],
                    peg_in_tool_m=[0,0,-.004],peg_quat_in_tool=[1,0,0,0],
                    peg_root_w_m=[0,0,.02],tool_tip_w_m=[0,0,.024],
                    tool_quaternion_wxyz=[1,0,0,0],
                    finger_contact_points_w_m=[[-.002,0,.02],[.002,0,.02]],
                    finger_contact_surfaces=['outer','outer'])

    def test_wall_keeps_original_controller_and_candidate(self):
        self.assertIs(type(grip_controller('wall',.15)),GripController)
        plan=contact_candidate(load_profile(),'wall',2,.5,1)
        self.assertEqual(plan['entry_tilt_deg'],20.)
        self.assertEqual(plan['open_angle_rad'],.15)
        self.assertNotIn('pinch_center_offset_m',plan)

    def test_mesh_centering_reduces_first_contact_asymmetry(self):
        profile=load_profile()
        for arm in (1,2):
            plan=contact_candidate(profile,'vertex',2 if arm==1 else 0,.5,arm)
            self.assertTrue(preflight(profile,plan)['sampled_arrival_clear'])
            self.assertLessEqual(plan['closure_geometry_check']['angle_imbalance_rad'],.005+1e-9)
            self.assertLessEqual(abs(plan['pinch_center_offset_m']),.001)
        centered=contact_candidate(profile,'vertex',2,.5,1)
        uncentered=copy.deepcopy(centered)
        axis=rotation(centered['tool_quaternion_local_wxyz']).as_matrix()[:,0]
        uncentered['tip_local_m']=(np.asarray(centered['tip_local_m'])-axis*centered['pinch_center_offset_m']).tolist()
        self.assertGreater(closure_contact_screen(profile,uncentered)['angle_imbalance_rad'],.04)

    def test_first_contact_pauses_closure_and_centers_toward_loaded_finger(self):
        for a,b,sign in ((0.,.04,1),(.04,0.,-1)):
            controller=grip_controller('vertex',.5)
            for _ in range(100):result=controller.update(self.row(a,b),'close',.01)
            self.assertTrue(result['jaw_closure_paused'])
            self.assertAlmostEqual(result['jaw_target_rad'],.5)
            self.assertGreater(sign*result['centering_position_w_m'][0],0.)
            self.assertLessEqual(abs(result['centering_offset_mm']),1.)
            self.assertFalse(result['pressure_ready'])

    def test_coupled_jaws_and_centering_load_an_offset_contact_plant(self):
        # Toy compliant plant checks the one-sided-contact feedback, not Isaac
        # friction, real collision normals, or physical grasp success.
        controller=grip_controller('vertex',.5)
        center=0.;angle=.5;peak=0.;ready=False
        for _ in range(1000):
            closure=(.4-angle)*.009
            force=np.maximum(0.,np.array([closure-.0003+center,closure+.0003-center])*1000)
            row=self.row(*force);row['tool_tip_w_m'][0]=center;row['peg_in_tool_m'][0]=-center
            result=controller.update(row,'close',.01)
            self.assertIsNone(result['reason'])
            center=result['centering_position_w_m'][0];angle=result['jaw_target_rad']
            peak=max(peak,force.max());ready|=result['pressure_ready']
        self.assertTrue(ready);self.assertGreaterEqual(force.min(),.30)
        self.assertLess(peak,.55)

    def test_measured_vertex_ejection_is_stopped_before_lift(self):
        samples=json.loads(Path(__file__).with_name('measured_vertex_closure_samples.json').read_text())['samples']
        controller=grip_controller('vertex',.5)
        controller.update(samples[0],'close',.01)
        first=controller.update(samples[1],'close',.01)
        self.assertTrue(first['jaw_closure_paused']);self.assertFalse(first['pressure_ready'])
        terminal=controller.update(samples[-1],'close',.01)
        self.assertEqual(terminal['reason'],'vertex_ejected_during_close')
        self.assertGreater(terminal['closure_peg_displacement_mm'],2.)
        self.assertFalse(terminal['motion_allowed']);self.assertFalse(terminal['pressure_ready'])

    def test_tool_centering_is_not_reported_as_object_ejection(self):
        controller=grip_controller('vertex',.5)
        controller.update(self.row(),'close',.01)
        row=self.row(.1,.1);row['tool_tip_w_m'][0]=.0009;row['peg_in_tool_m'][0]=-.0009
        result=controller.update(row,'close',.01)
        self.assertIsNone(result['reason']);self.assertEqual(result['closure_peg_displacement_mm'],0.)
        self.assertFalse(result['pressure_ready'])

    def test_balance_timeout_and_excessive_force_prevent_lift(self):
        controller=grip_controller('vertex',.5)
        for _ in range(301):result=controller.update(self.row(0.,.1),'close',.01)
        self.assertEqual(result['reason'],'vertex_contact_balance_not_recovered')
        self.assertFalse(result['pressure_ready'])
        result=grip_controller('vertex',.5).update(self.row(1.01,0.),'close',.01)
        self.assertEqual(result['reason'],'excessive_finger_force')
        self.assertFalse(result['pressure_ready'])


class KinematicsTests(unittest.TestCase):
    def test_jacobian_matches_forward_kinematics_derivatives(self):
        model=KinematicModel()
        q=np.array([.4,.1,.12,.45,-.9,-.2])
        analytic=model.jacobian(q);pose=model.forward(q);eps=1e-7
        for i in range(6):
            trial=q.copy();trial[i]+=eps
            next_pose=model.forward(trial)
            numerical=np.r_[(next_pose[:3,3]-pose[:3,3])/eps,
                (Rotation.from_matrix(next_pose[:3,:3])*Rotation.from_matrix(pose[:3,:3]).inv()).as_rotvec()/eps]
            np.testing.assert_allclose(analytic[:,i],numerical,atol=2e-7)

    def test_actual_sdk_dls_tracks_oblique_approaches_with_servo_lag(self):
        # Real SDK IK + measured USD chain + a simple lagging joint plant. This
        # catches IK branch/limit stalls; it does not simulate contact physics.
        import torch
        from isaaclab.controllers.differential_ik import DifferentialIKController
        model=KinematicModel();profile=load_profile()
        root=np.array([-profile['hole_center_m'][0],0,.010-profile['bounds_m'][0][2]])
        base=np.array([-.07,0,.15])
        cfg=SimpleNamespace(command_type='pose',use_relative_mode=False,ik_method='dls',ik_params={'lambda_val':.01})
        for mode in ('wall','vertex'):
            plan=contact_candidate(profile,mode,2,.5,1)
            goal=root+plan['tip_local_m']+.015*np.asarray(plan['approach_direction_local'])
            goal[2]=max(root[2]+plan['tip_local_m'][2]+.025,.055)
            q=np.array([.01,.01,.07,.01,.01,.01])
            pose=model.forward(q)
            quat=Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]]
            path=PoseTrajectory(base+pose[:3,3],quat)
            ik=DifferentialIKController(cfg,1,'cpu')
            speed_limits=np.array([1.5,1.5,.04,1.5,1.5,1.5])
            for _ in range(400):
                pose=model.forward(q);position=base+pose[:3,3]
                quat=Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]]
                command,commandq=path.advance(position,quat,goal,plan['tool_quaternion_local_wxyz'],.01,.025,1.)
                ik.set_command(torch.tensor(np.r_[command-base,commandq][None],dtype=torch.float32))
                desired=ik.compute(torch.tensor(pose[:3,3][None],dtype=torch.float32),
                    torch.tensor(quat[None],dtype=torch.float32),torch.tensor(model.jacobian(q)[None],dtype=torch.float32),
                    torch.tensor(q[None],dtype=torch.float32))[0].numpy()
                velocity=np.clip(20*(desired-q),-speed_limits,speed_limits)
                q=np.clip(q+.01*velocity,model.limits[:,0],model.limits[:,1])
            pose=model.forward(q)
            self.assertLess(np.linalg.norm(goal-base-pose[:3,3]),.0007,mode)
            self.assertLess(np.linalg.norm((rotation(plan['tool_quaternion_local_wxyz'])*Rotation.from_matrix(pose[:3,:3]).inv()).as_rotvec()),.05,mode)

    def test_usd_forward_kinematics_matches_measured_simulation_poses(self):
        model=KinematicModel()
        samples=json.loads(Path(__file__).with_name('measured_kinematics_samples.json').read_text())
        for sample in samples:
            pose=model.forward(sample['joint_positions'])
            position=pose[:3,3]+sample['base_position_w_m']
            self.assertLess(np.linalg.norm(position-sample['tool_tip_w_m']),1e-6)
            error=(Rotation.from_matrix(pose[:3,:3])*rotation(sample['tool_quaternion_wxyz']).inv()).as_rotvec()
            self.assertLess(np.linalg.norm(error),2e-6)

    def test_previous_vertical_and_horizontal_poses_are_rejected(self):
        profile=load_profile()
        for mode in ('wall','vertex'):
            old=candidate(profile,mode,vertex=2)
            old['arm']=1
            screen=route_check(profile,old)
            self.assertFalse(screen['reachable'])
            self.assertEqual(screen['failed_stage'],'above')

    def test_oblique_routes_are_feasible_for_both_arms_with_original_limits(self):
        profile=load_profile();model=KinematicModel()
        original_limits=model.limits.copy()
        for arm in (1,2):
            for mode in ('wall','vertex'):
                plan=contact_candidate(profile,mode,2 if arm==1 else 0,.5,arm)
                self.assertTrue(preflight(profile,plan)['sampled_arrival_clear'],(arm,mode))
                report=route_check(profile,plan)
                self.assertTrue(report['reachable'],(arm,mode,report))
                self.assertEqual(report['samples_checked'],72)
                self.assertEqual({r['stage'] for r in report['route_samples']},
                                 {'above','outside','reach','lift','lower','retreat'})
                base=np.array([-.07 if arm==1 else .07,0,.15])
                for row in report['route_samples']:
                    q=np.array([row['joints'][name] for name in model.names])
                    self.assertTrue((q>=model.bounds[:,0]-1e-9).all())
                    self.assertTrue((q<=model.bounds[:,1]+1e-9).all())
                    measured=model.forward(q)
                    self.assertLess(np.linalg.norm(base+measured[:3,3]-row['position_w_m']),.0002)
                    self.assertLess(np.linalg.norm((Rotation.from_matrix(measured[:3,:3])*rotation(row['quaternion_wxyz']).inv()).as_rotvec()),.02)
        np.testing.assert_array_equal(model.limits,original_limits)

    def test_kinematics_for_another_robot_asset_are_rejected(self):
        profile=copy.deepcopy(load_profile());profile['robot']['asset_sha256']='changed'
        plan=contact_candidate(profile,'wall',2,.5,1)
        with self.assertRaisesRegex(ValueError,'asset hash'):
            route_check(profile,plan)


class MotionTests(unittest.TestCase):
    def test_pressure_pause_stops_reference_and_resumes_without_backlog(self):
        trajectory=PoseTrajectory([0,0,0],[1,0,0,0])
        actual=np.zeros(3);quat=[1,0,0,0];goal=[0,0,.035]
        for _ in range(20):
            command,_=trajectory.advance(actual,quat,goal,quat,.01,.003,.2,True)
            actual=command
        stopped=actual.copy()
        for _ in range(100):
            command,_=trajectory.advance(actual,quat,goal,quat,.01,.003,.2,True,False)
            np.testing.assert_array_equal(command,stopped)
        actual=stopped+np.array([0,0,-.0002])
        command,_=trajectory.advance(actual,quat,goal,quat,.01,.003,.2,True,True)
        self.assertLessEqual(np.linalg.norm(command-actual),.003*.01+1e-10)

    def test_persistent_reference_overcomes_servo_lag_without_jump(self):
        # A lagging plant, not perfect kinematics. This checks tracking/reference
        # behavior only; force and real PSM dynamics require user-run Isaac tests.
        position=np.zeros(3);quat=np.array([1.,0,0,0])
        goal=np.array([.09,.01,.02])
        goal_quat=Rotation.from_rotvec([.2,.8,.6]).as_quat()[[3,0,1,2]]
        trajectory=PoseTrajectory(position,quat)
        previous=position.copy();previous_quat=quat.copy()
        for step in range(600):
            command,command_quat=trajectory.advance(position,quat,goal,goal_quat,.01,.025,1.)
            self.assertLessEqual(np.linalg.norm(command-position),.003+1e-10)
            self.assertLessEqual(np.linalg.norm((rotation(command_quat)*rotation(quat).inv()).as_rotvec()),.15+1e-10)
            self.assertLessEqual(np.linalg.norm(command-previous),.025*.01+1e-10)
            self.assertLessEqual(np.linalg.norm((rotation(command_quat)*rotation(previous_quat).inv()).as_rotvec()),.01+1e-10)
            previous=command;previous_quat=command_quat
            position += .2*(command-position)
            delta=(rotation(command_quat)*rotation(quat).inv()).as_rotvec()
            quat=(Rotation.from_rotvec(.2*delta)*rotation(quat)).as_quat()[[3,0,1,2]]
        self.assertLess(np.linalg.norm(goal-position),.0007)
        self.assertLess(np.linalg.norm((rotation(goal_quat)*rotation(quat).inv()).as_rotvec()),.05)

    def test_obstructed_motion_cannot_wind_up_and_contact_has_smaller_lead(self):
        for contact,maximum in [(False,.003),(True,.0005)]:
            trajectory=PoseTrajectory([0,0,0],[1,0,0,0])
            for _ in range(1000):
                command,quat=trajectory.advance([0,0,0],[1,0,0,0],[1,0,0],[0,0,0,1],.01,.005,1.,contact)
                self.assertLessEqual(np.linalg.norm(command),maximum+1e-10)
                self.assertLessEqual(np.linalg.norm(rotation(quat).as_rotvec()),(.04 if contact else .15)+1e-10)

    def test_small_waypoint_error_does_not_bypass_orientation_or_contact(self):
        self.assertFalse(pose_reached(.28,.116))
        watchdog=MotionWatchdog()
        for step in range(500):
            self.assertFalse(watchdog.stalled(step*.01,.28,.116))
        self.assertTrue(watchdog.stalled(5.,.28,.116))
        self.assertFalse(MotionWatchdog().stalled(5.,.28,.04))
        watchdog=MotionWatchdog()
        for step in range(1001):
            self.assertFalse(watchdog.stalled(step*.01,90-step*.025,1-step*.0005))

    def test_wall_zero_roll_arrival_preserves_clearance_on_both_branches(self):
        profile=load_profile();plan=candidate(profile,mode='wall')
        np.testing.assert_allclose(plan['tool_quaternion_local_wxyz'],[1,0,0,0])
        self.assertTrue(preflight(profile,plan)['sampled_arrival_clear'])
        plan['tool_quaternion_local_wxyz']=[0,0,0,1]
        self.assertTrue(preflight(profile,plan)['sampled_arrival_clear'])

    def test_absolute_action_uses_real_isaac_ik_frame_and_dimensions(self):
        import torch
        # Load the actual SDK math/controller without the simulator or its GUI.
        repo=Path(__file__).resolve().parents[2]
        sdk=repo/'third_party/IsaacLab/source/isaaclab/isaaclab'
        spec=importlib.util.spec_from_file_location('grasp_sdk_math',sdk/'utils/math.py')
        math_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(math_module)
        source=ast.parse(Path(__file__).with_name('runtime.py').read_text())
        function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='control')
        scope={'torch':torch,'subtract_frame_transforms':math_module.subtract_frame_transforms}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'runtime_control','exec'),scope)
        roots=[torch.tensor([[-.07,0.,.15]]),torch.tensor([[.07,0.,.15]])]
        root_quat=torch.tensor([[np.sqrt(.5),0.,0.,np.sqrt(.5)]],dtype=torch.float32)
        tip=torch.tensor([[.01,.02,.055]])
        world_quat=torch.tensor([[1.,0,0,0]])
        scene={}
        for arm in (1,2):
            scene[f'robot_{arm}']=SimpleNamespace(data=SimpleNamespace(root_pos_w=roots[arm-1],root_quat_w=root_quat))
            scene[f'ee_{arm}_frame']=SimpleNamespace(data=SimpleNamespace(target_pos_w=tip[:,None],target_quat_w=world_quat[:,None]))
        env=SimpleNamespace(device='cpu',scene=scene)
        for active in (1,2):
            target=tip+torch.tensor([[.001,0,0]])
            action=scope['control'](env,target,world_quat,.15,active)
            self.assertEqual(tuple(action.shape),(1,16))
            for arm in (1,2):
                offset=(arm-1)*8
                actual_world,_=math_module.combine_frame_transforms(roots[arm-1],root_quat,action[:,offset:offset+3],action[:,offset+3:offset+7])
                torch.testing.assert_close(actual_world,target if arm==active else tip)
                self.assertAlmostEqual(float(action[0,offset+7]),-.4 if arm==active else 1.,places=6)
        from isaaclab.controllers.differential_ik import DifferentialIKController
        cfg=SimpleNamespace(command_type='pose',use_relative_mode=False,ik_method='dls',ik_params={'lambda_val':.01})
        controller=DifferentialIKController(cfg,1,'cpu')
        self.assertEqual(controller.action_dim,7)
        local_pos,local_quat=math_module.subtract_frame_transforms(roots[1],root_quat,tip,world_quat)
        controller.set_command(action[:,8:15],local_pos,local_quat)
        joints=controller.compute(local_pos,local_quat,torch.eye(6)[None],torch.zeros(1,6))
        self.assertTrue(torch.isfinite(joints).all())
        self.assertGreater(float(joints.norm()),0.)


class TimingTests(unittest.TestCase):
    def test_longer_phase_budget_does_not_lengthen_required_grasp_hold(self):
        limits=dict(stage_limits(6000,.01))
        self.assertEqual(limits['settle']*.01,.5)
        self.assertEqual(limits['above']*.01,60.)
        self.assertEqual(limits['reach']*.01,60.)
        self.assertEqual(limits['release']*.01,60.)
        self.assertLess(hold_steps(.01),limits['hold'])
        self.assertEqual(hold_steps(.01)*.01,2.)
        self.assertEqual(hold_steps(.02)*.02,2.)

    def test_invalid_timing_is_rejected(self):
        for steps,dt in [(0,.01),(6000,0.),(6000,float('nan'))]:
            with self.assertRaises(ValueError):stage_limits(steps,dt)


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile=load_profile()
        cls.hulls=[ConvexHull(cell).equations for cell in sectors(cls.profile)]

    def inside(self,point):
        return any(np.all(h[:,:3]@point+h[:,3]<1e-9) for h in self.hulls)

    def test_hole_and_board_post_stay_open_at_all_layers(self):
        p=self.profile
        cx,cy=p['hole_center_m']
        heights=[l['z_m'] for l in p['levels']]
        heights += [(a+b)/2 for a,b in zip(heights,heights[1:])]
        for z in heights:
            for a in np.linspace(0,2*np.pi,32):
                self.assertFalse(self.inside([cx+.002*np.cos(a),cy+.002*np.sin(a),z]))

    def test_rectangular_ghost_corners_are_removed_and_wall_remains(self):
        z=-.006
        self.assertFalse(self.inside([.014,.008,z]))
        self.assertFalse(self.inside([.014,-.008,z]))
        self.assertTrue(self.inside([0.,0.,z]))
        self.assertTrue(self.inside([.014,0.,z]))
        self.assertLess(self.profile['sampled_max_radial_error_mm'],.20)

    def test_rounded_lips_and_convex_cells_are_finite(self):
        levels=self.profile['levels']
        self.assertTrue(np.all(np.array(levels[0]['inner_radius_m'])>levels[3]['inner_radius_m']))
        self.assertTrue(np.all(np.array(levels[0]['outer_radius_m'])<levels[3]['outer_radius_m']))
        for hull in self.hulls:
            self.assertTrue(np.isfinite(hull).all())

    def test_asset_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'peg.usd';path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'differs'):
                verify_asset(self.profile,path)

    def test_candidates_use_three_vertices_and_inner_outer_wall(self):
        points=[candidate(self.profile,vertex=i)['tip_local_m'] for i in range(3)]
        self.assertTrue(points[0][0]>points[1][0])
        self.assertGreater(points[1][1],0)
        self.assertLess(points[2][1],0)
        wall=candidate(self.profile,mode='wall')
        self.assertTrue(.001<wall['wall_thickness_m']<.004)
        with self.assertRaises(ValueError):candidate(self.profile,height=0)

    def test_open_vertex_arrivals_clear_peg_post_and_board_at_three_heights(self):
        for vertex in range(3):
            for height in (.2,.5,.8):
                plan=candidate(self.profile,vertex=vertex,height=height)
                self.assertTrue(preflight(self.profile,plan)['sampled_arrival_clear'],(vertex,height))
                self.assertEqual(plan['approach'],'side_entry')

    def test_wall_arrival_can_straddle_wall_without_hitting_source_post(self):
        plan=candidate(self.profile,mode='wall')
        self.assertTrue(preflight(self.profile,plan)['sampled_arrival_clear'])
        plan['open_angle_rad']=.25
        report=preflight(self.profile,plan)
        self.assertFalse(report['sampled_arrival_clear'])
        self.assertTrue(any(report['jaw_surface_samples_inside_source_post']))


class GripTests(unittest.TestCase):
    def test_peg_initialization_updates_legacy_shapes_and_preserves_restitution(self):
        import torch
        source=ast.parse(Path(__file__).with_name('runtime.py').read_text())
        functions=[n for n in source.body if isinstance(n,ast.FunctionDef)
                   and n.name in ('material_evidence','initialize_contact_physics')]
        scope={'np':np,'torch':torch}
        exec(compile(ast.Module(body=functions,type_ignores=[]),'contact_initialization','exec'),scope)

        class MaterialView:
            def __init__(self,values,ignore_writes=False):
                self.values=torch.tensor([values]);self.count=1;self.writes=[];self.ignore_writes=ignore_writes
            def get_material_properties(self):
                return self.values.clone()
            def set_material_properties(self,values,ids):
                self.writes.append((values.clone(),ids.clone()))
                if not self.ignore_writes:self.values=values.clone()

        view=MaterialView([[.9,.9,-1000.],[.6,.5,-1000.],[.5,.5,0.]])
        robot=SimpleNamespace(root_physx_view=SimpleNamespace(max_shapes=4,link_paths=[['a','b']],
            get_material_properties=lambda:torch.tensor([[[.9,.9,0.]]*4])),
            _physics_sim_view=SimpleNamespace(create_rigid_body_view=lambda path:SimpleNamespace(max_shapes=2)),
            find_bodies=lambda name:([0 if 'gripper1' in name else 1],[]))
        scene={'robot_1':robot,'robot_2':robot,'object':SimpleNamespace(root_physx_view=view)}
        for arm in (1,2):
            for finger in (1,2):
                scene[f'contact_{arm}_{finger}']=SimpleNamespace(data=SimpleNamespace(force_matrix_w=np.zeros((1,1,1,3))))
        env=SimpleNamespace(scene=scene)
        before=view.values.clone()
        with redirect_stdout(io.StringIO()):
            scope['initialize_contact_physics'](env,.9)
        torch.testing.assert_close(view.values[...,:2],torch.full_like(view.values[...,:2],.9))
        torch.testing.assert_close(view.values[...,2],before[...,2])
        torch.testing.assert_close(view.writes[0][1],torch.tensor([0],dtype=torch.int32))
        self.assertEqual(view.writes[0][1].device.type,'cpu')
        self.assertEqual(env.grasp_material_evidence['peg']['shapes'],3)
        self.assertEqual(len(env.grasp_material_initial_evidence['peg_coefficients']),3)
        # Correct USD bindings need no further PhysX write.
        with redirect_stdout(io.StringIO()):
            scope['initialize_contact_physics'](env,.9)
        self.assertEqual(len(view.writes),1)
        scene['object'].root_physx_view=MaterialView([[.6,.6,0.]],ignore_writes=True)
        with self.assertRaisesRegex(RuntimeError,'Peg actual friction differs.*coefficients='):
            scope['initialize_contact_physics'](env,.9)

    def test_environment_is_closed_when_contact_initialization_fails(self):
        import probe
        calls=[]
        env=SimpleNamespace(close=lambda:calls.append('env_close'),
                            cfg=SimpleNamespace(viewer=SimpleNamespace(eye=None,lookat=None)),
                            viewport_camera_controller=None,
                            grasp_material_initial_evidence={'peg_coefficients':[[.6,.6]]})
        app=SimpleNamespace(close=lambda:calls.append('app_close'))
        def fail(env,friction):
            raise RuntimeError('initialization regression')
        runtime=SimpleNamespace(make_env=lambda *args:env,initialize_contact_physics=fail,
                                control=None,telemetry=None)
        fake_launcher=SimpleNamespace(AppLauncher=lambda args:SimpleNamespace(app=app))
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'startup_failure.json'
            args=['probe.py','--headless','--episodes','1','--output',str(output)]
            with mock.patch.dict(sys.modules,{'isaaclab.app':fake_launcher,'runtime':runtime}), \
                 mock.patch.object(sys,'argv',args), \
                 mock.patch.object(probe,'route_check',return_value={'reachable':True,'samples_checked':72}), \
                 redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()) as errors:
                self.assertEqual(probe.main(),1)
            self.assertIn('initialization regression',errors.getvalue())
            report=json.loads(output.read_text())
            self.assertFalse(report['complete']);self.assertEqual(report['episodes'],0)
            self.assertEqual(report['initial_physx_materials'],env.grasp_material_initial_evidence)
            self.assertEqual(calls,['env_close','app_close'])

    def row(self,a=.35,b=.35):
        return dict(finger_normal_force_n=[a,b],finger_force_vectors_n=[[a,0,0],[-b,0,0]],
                    peg_in_tool_m=[0,0,-.004],peg_quat_in_tool=[1,0,0,0],
                    finger_contact_points_w_m=[[0,0,0],[.003,0,0]],
                    finger_contact_surfaces=['inner','outer'])

    def loaded(self):
        controller=GripController('wall',.15)
        for _ in range(40):result=controller.update(self.row(),'close',.01)
        self.assertTrue(result['pressure_ready'])
        return controller

    def test_measured_failed_grasps_do_not_pass_preload(self):
        samples=json.loads(Path(__file__).with_name('measured_grip_samples.json').read_text())
        for mode,fixture in samples.items():
            controller=GripController(mode,.15 if mode=='wall' else .45)
            result=None
            for _ in range(1200):
                result=controller.update(fixture['sample'],'close',.01)
                self.assertFalse(result['pressure_ready'])
                self.assertFalse(result['motion_allowed'])
                if result['reason']:break
            self.assertEqual(result['reason'],'jaw_closed_without_secure_contact')

    def test_preload_requires_pressure_on_both_fingers_and_stable_position(self):
        controller=GripController('wall',.15)
        for step in range(39):
            self.assertFalse(controller.update(self.row(),'close',.01)['pressure_ready'])
        disturbed=self.row();disturbed['peg_in_tool_m'][0]=.0008
        self.assertFalse(controller.update(disturbed,'close',.01)['pressure_ready'])
        for step in range(40):result=controller.update(disturbed,'close',.01)
        self.assertTrue(result['pressure_ready'])
        for invalid in (self.row(.35,0.),self.row(.05,.05)):
            trial=GripController('wall',.15)
            for _ in range(100):self.assertFalse(trial.update(invalid,'close',.01)['pressure_ready'])

    def test_lift_pauses_on_weak_force_and_resumes_only_after_recovery(self):
        controller=self.loaded()
        self.assertTrue(controller.update(self.row(),'lift',.01)['motion_allowed'])
        weak=self.row(.10,.45)
        before=controller.angle
        result=controller.update(weak,'lift',.01)
        self.assertFalse(result['motion_allowed'])
        self.assertLess(result['jaw_target_rad'],before)
        for _ in range(9):self.assertFalse(controller.update(self.row(),'lift',.01)['motion_allowed'])
        self.assertTrue(controller.update(self.row(),'lift',.01)['motion_allowed'])

    def test_slip_loss_and_force_limit_abort_instead_of_empty_lift(self):
        controller=self.loaded();slipped=self.row();slipped['peg_in_tool_m'][2]-=.0016
        result=controller.update(slipped,'lift',.01)
        self.assertEqual(result['reason'],'grasp_slipped');self.assertFalse(result['motion_allowed'])
        controller=self.loaded()
        for _ in range(40):result=controller.update(self.row(0,0),'lift',.01)
        self.assertEqual(result['reason'],'finger_contact_lost')
        controller=self.loaded()
        for _ in range(100):result=controller.update(self.row(.15,.15),'lift',.01)
        self.assertEqual(result['reason'],'grip_pressure_not_recovered')
        controller=self.loaded();result=controller.update(self.row(1.01,.35),'lift',.01)
        self.assertEqual(result['reason'],'excessive_finger_force')
        self.assertFalse(result['motion_allowed'])

    def test_nominal_capacity_is_only_screen_and_invalid_force_target_is_rejected(self):
        with self.assertRaises(ValueError):GripController('wall',.15,target_force=.98)
        with self.assertRaises(ValueError):GripController('wall',.15,mass_kg=.1)
        result=GripController('wall',.15).update(self.row(),'lift',.01)
        self.assertGreater(result['nominal_friction_capacity_n'],result['peg_weight_n'])
        self.assertEqual(result['reason'],'missing_pressure_preload')
        self.assertFalse(result['motion_allowed'])

    def test_pressure_feedback_loads_a_lagging_contact_without_force_spike(self):
        # A compliant, lagging pressure plant checks controller transients only.
        # It cannot establish frictional grasp success in Isaac.
        controller=GripController('wall',.15)
        force=0.;peak=0.;ready=False
        for _ in range(600):
            result=controller.update(self.row(force,force),'close',.01)
            self.assertIsNone(result['reason'])
            expected=max(0.,(.145-result['jaw_target_rad'])*.6/.009)
            force+=.25*(expected-force)
            peak=max(peak,force)
            ready|=result['pressure_ready']
        self.assertTrue(ready)
        self.assertGreaterEqual(force,.30)
        self.assertLess(peak,.45)

    def test_cooked_finger_and_peg_friction_are_verified(self):
        import torch
        source=ast.parse(Path(__file__).with_name('runtime.py').read_text())
        function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='material_evidence')
        scope={'np':np}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'material_evidence','exec'),scope)
        values=torch.tensor([[[.9,.9,0.]]*4])
        robot=SimpleNamespace(root_physx_view=SimpleNamespace(max_shapes=4,link_paths=[['a','b']],
            get_material_properties=lambda:values),
            _physics_sim_view=SimpleNamespace(create_rigid_body_view=lambda path:SimpleNamespace(max_shapes=2)),
            find_bodies=lambda name:([0 if 'gripper1' in name else 1],[]))
        peg_values=torch.tensor([[[.9,.9,0.]]*3])
        peg=SimpleNamespace(root_physx_view=SimpleNamespace(get_material_properties=lambda:peg_values))
        env=SimpleNamespace(scene={'robot_1':robot,'robot_2':robot,'object':peg})
        evidence=scope['material_evidence'](env,.9)
        self.assertEqual(evidence['psm1_finger2']['shapes'],2)
        self.assertEqual(evidence['peg']['shapes'],3)
        values[0,2,1]=.5
        with self.assertRaisesRegex(RuntimeError,'actual friction differs'):
            scope['material_evidence'](env,.9)
        values[0,2,1]=.9;peg_values[0,0,0]=.6
        with self.assertRaisesRegex(RuntimeError,'Peg actual friction differs'):
            scope['material_evidence'](env,.9)


class ContactEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.row=dict(finger_force_vectors_n=[[.10,0,0],[-.10,0,0]],peg_bottom_m=.050,
            peg_speed_m_s=.001,peg_in_tool_m=[0,0,-.004],
            finger_contact_points_w_m=[[0,0,.030],[.003,0,.030]],finger_contact_surfaces=['inner','outer'])

    def test_opposed_contact_plus_lift_and_no_slip(self):
        self.assertTrue(physical_hold(self.row,.010,[0,0,-.004]))
        for field,value in [('finger_force_vectors_n',[[.1,0,0],[0,0,0]]),
                            ('finger_force_vectors_n',[[.1,0,0],[.1,0,0]]),
                            ('peg_bottom_m',.020),('peg_speed_m_s',.1),('peg_in_tool_m',[0,0,-.010])]:
            row=copy.deepcopy(self.row);row[field]=value
            self.assertFalse(physical_hold(row,.010,[0,0,-.004]),field)

    def test_wall_requires_one_inner_and_one_outer_contact(self):
        self.assertTrue(valid_contact_pair(self.row,'wall'))
        self.assertFalse(valid_contact_pair(self.row,'vertex'))
        self.row['finger_contact_surfaces']=['outer','outer']
        self.assertTrue(valid_contact_pair(self.row,'vertex'))
        self.assertFalse(valid_contact_pair(self.row,'wall'))
        self.row['finger_contact_points_w_m'][0]=None
        self.assertFalse(valid_contact_pair(self.row,'vertex'))

    def test_contact_surface_classification(self):
        p=load_profile();z=-.006
        # theta=0: inner and outer radii at a sampled height are measured from the hole center.
        center=np.array(p['hole_center_m'])
        for key,label in [('inner_radius_m','inner'),('outer_radius_m','outer')]:
            point=[center[0]+p['levels'][3][key][0],center[1],z]
            self.assertEqual(surface_label(p,point),label)

    def test_runtime_has_only_default_reset_and_unassisted_env(self):
        source=Path(__file__).with_name('runtime.py').read_text()
        tree=ast.parse(source)
        calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('write_root_pose_to_sim',calls)
        self.assertNotIn('write_root_velocity_to_sim',calls)
        self.assertNotIn('write_nodal_state_to_sim',calls)
        self.assertNotIn('mode=\'interval\'',source)
        self.assertIn('env = ManagerBasedEnv(cfg)',source)


if __name__=='__main__':
    unittest.main(verbosity=2)
