"""CUDA motion limits, matched evaluation gates and protected BC fine-tuning."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from autonomy_env import ContactMotorReference,AutonomyEnv
from autonomy_learning import load_policy,SCHEMA
from autonomy_motion_curriculum import distribution_gate,speed_comparison,sha
from autonomy_position_fit import approach_samples,fit
from rl_core import quaternion,rotvec,multiply,conjugate

ROOT=Path(__file__).resolve().parents[2]
class MotionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        cls.device='cuda:0';torch.set_num_threads(4)
    def test_speed_scales_translation_rotation_without_changing_jaws(self):
        d=self.device
        s=dict(tip=torch.tensor([[[0.,0.,.05]]*2],device=d),tool_q=quaternion(torch.zeros(1,2,3,device=d)),
            jaws=torch.ones(1,2,2,device=d)*.15,forces=torch.zeros(1,2,2,3,device=d))
        a=torch.tensor([[1.,0.,0.,0.,0.,1.,-1.]*2],device=d)
        for in_contact in (False,True):
            s['forces'].zero_()
            if in_contact:s['forces'][...,0]=.1
            old=ContactMotorReference(s);new=ContactMotorReference(s,1.5,1.25)
            old.advance(a,s);new.advance(a,s)
            factor=1.25 if in_contact else 1.5
            torch.testing.assert_close(new.p-s['tip'],(old.p-s['tip'])*factor,atol=1e-7,rtol=1e-4)
            qo=rotvec(multiply(old.q,conjugate(s['tool_q'])))
            qn=rotvec(multiply(new.q,conjugate(s['tool_q'])))
            torch.testing.assert_close(qn,qo*factor,atol=1e-7,rtol=1e-4)
            torch.testing.assert_close(new.jaw,old.jaw)
        for value in (float('nan'),float('inf'),.9,2.1):
            with self.assertRaises(ValueError):ContactMotorReference(s,value,1.)
    def test_matched_speed_gate_rejects_lost_case_even_with_equal_rate(self):
        rows=[dict(success=i<15,length=5000,peak_force_n=.5,
            reset=dict(source='L5',target='R2',jitter_mm=[.1,-.1],yaw_deg=.5)) for i in range(16)]
        reference=dict(policy_sha256='same',seed=123,cases=rows)
        candidate=copy.deepcopy(reference)
        for c in candidate['cases']:c['length']=4000
        self.assertTrue(speed_comparison(reference,candidate))
        candidate['cases'][0]['success']=False;candidate['cases'][-1]['success']=True
        self.assertFalse(speed_comparison(reference,candidate))
        candidate=copy.deepcopy(reference)
        self.assertFalse(speed_comparison(reference,candidate))
        candidate['cases'][0]['reset']['yaw_deg']=-.5
        self.assertFalse(speed_comparison(reference,candidate))
    def test_reset_moves_hole_about_same_post_and_preserves_other_environment(self):
        from types import SimpleNamespace
        from rl_core import apply
        d=self.device
        env=AutonomyEnv.__new__(AutonomyEnv)
        class Object:
            def __init__(obj):obj.data=SimpleNamespace(root_state_w=torch.zeros(2,13,device=d));obj.data.root_state_w[:,3]=1
            def write_root_pose_to_sim(obj,pose,env_ids):obj.data.root_state_w[env_ids,:7]=pose
            def write_root_velocity_to_sim(obj,velocity,env_ids):obj.velocity=velocity
        class Scene(dict):
            def update(scene,dt):pass
        obj=Object();scene=Scene(object=obj)
        env.env=SimpleNamespace(scene=scene);env.device=d
        env.source_values=torch.tensor([[0.,0.]],device=d);env.target_values=torch.tensor([[.042,.015]],device=d)
        env.source_names=['L5'];env.target_names=['R2'];env.source=torch.tensor([[0.,0.,.01]]*2,device=d)
        env.target=env.source.clone();env.origins=torch.tensor([[0.,0.,0.],[2.5,0.,0.]],device=d)
        env.profile=json.loads((ROOT/'tools/peg_grasp_v2/peg_profile.json').read_text())
        env.jitter=.001;env.yaw=5*torch.pi/180;env.reset_rng=torch.Generator(device=d).manual_seed(123)
        env.reset_records=[None,None]
        untouched=obj.data.root_state_w[1].clone()
        env.randomize(torch.tensor([0],device=d))
        center=torch.tensor([*env.profile['hole_center_m'],0.],device=d)
        pose=obj.data.root_state_w[0,:7];hole=pose[:3]+apply(pose[3:],center)
        expected=env.source[0,:2]+torch.tensor(env.reset_records[0]['jitter_mm'],device=d)*.001
        torch.testing.assert_close(hole[:2],expected,atol=1e-8,rtol=1e-5)
        torch.testing.assert_close(obj.data.root_state_w[1],untouched)
        self.assertTrue(env.reset_records[0]['board_and_posts_unchanged'])
        # Even a square-corner offset keeps the cylindrical post inside the hole.
        radius=min(min(v['inner_radius_m']) for v in env.profile['levels'])
        self.assertGreater(radius-.002-(2**.5)*env.jitter,.0001)
    def report(self):
        path=ROOT/'logs/grasp_v2/autonomy_full_constraints_curriculum_v41/support_1p0_seed123/evaluation.json'
        r=json.loads(path.read_text())
        r['reset_distribution']=dict(sources=['L5'],targets=['R2'],jitter_mm=.25,yaw_deg=1.,reset_only=True)
        for c in r['cases']:
            c['reset'].update(source_post_center_m=[0.,0.,.01],reset_only_randomization=True,board_and_posts_unchanged=True)
        return r
    def test_distribution_gate_requires_same_post_bounded_offsets_and_real_full_success(self):
        r=self.report();self.assertTrue(distribution_gate(r,.25,1.))
        r['cases'][0]['reset']['jitter_mm'][0]=.251
        self.assertFalse(distribution_gate(r,.25,1.))
        r=self.report();r['cases'][0]['reset']['source_post_center_m'][0]=.02
        self.assertFalse(distribution_gate(r,.25,1.))
        r=self.report();r['teacher_at_inference']=True
        self.assertFalse(distribution_gate(r,.25,1.))
    def dataset(self,folder):
        parent=ROOT/'logs/grasp_v2/autonomy_full_firm_constraints_v41'
        net=load_policy(parent/'policy.pt',self.device)
        x=net.mean.repeat(32,1).cpu().numpy();x[:,0]+=.01
        episode=np.repeat(np.arange(8),4);validation=episode>=6
        meta=dict(schema=SCHEMA,skill='full',collection_complete=True,assisted_grasp=True,episodes=list(range(8)))
        path=folder/'demonstrations.npz'
        np.savez_compressed(path,obs=x,stage=np.tile(np.arange(4),8),episode=episode,validation=validation,metadata=json.dumps(meta))
        report=folder/'evaluation.json';report.write_text(json.dumps(dict(complete=True,assisted_grasp=True,
            cases=[dict(success=True,full_success=True)]*8)))
        (folder/'environment_close.json').write_text(json.dumps(dict(environment_closed=True,evaluation_sha256=sha(report))))
        (folder/'manifest.json').write_text((parent/'manifest.json').read_text())
        return parent,path
    def test_assisted_training_cannot_masquerade_as_native_or_mix_episode_splits(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as f:
            folder=Path(f);_,path=self.dataset(folder)
            arrays=dict(np.load(path,allow_pickle=False));meta=json.loads(str(arrays['metadata']))
            meta['assisted_grasp']=False;arrays['metadata']=json.dumps(meta)
            np.savez_compressed(path,**arrays)
            with self.assertRaisesRegex(ValueError,'Explicitly assisted'):approach_samples(path,self.device)
            meta['assisted_grasp']=True;arrays['metadata']=json.dumps(meta);arrays['validation'][0]=True
            np.savez_compressed(path,**arrays)
            with self.assertRaisesRegex(ValueError,'Separate whole'):approach_samples(path,self.device)
    def test_fine_tuning_preserves_gates_jaws_contact_and_other_motor_rows(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as f:
            folder=Path(f);parent,path=self.dataset(folder)
            evidence=fit(parent/'policy.pt',path,folder/'candidate',self.device,2)
            self.assertTrue(evidence['assisted_training']);self.assertTrue(evidence['physical_evaluation_required'])
            self.assertTrue(evidence['changed_tensors'])
            self.assertTrue(all(k.startswith('pickup_decoder.pose_head.') for k in evidence['changed_tensors']))
            before=load_policy(parent/'policy.pt',self.device).state_dict()
            after=load_policy(folder/'candidate/policy.pt',self.device).state_dict()
            for key,value in before.items():
                if not key.startswith('pickup_decoder.pose_head.'):self.assertTrue(torch.equal(value,after[key]),key)
            for start in (7,21,35,42):
                self.assertTrue(torch.equal(before['pickup_decoder.pose_head.weight'][start:start+7],after['pickup_decoder.pose_head.weight'][start:start+7]))

if __name__=='__main__':unittest.main(verbosity=2)
