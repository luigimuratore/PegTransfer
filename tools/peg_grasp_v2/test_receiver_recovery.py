"""CUDA regressions for supervised handoff and strictly isolated receiver BC."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from autonomy_learning import load_policy,SCHEMA
from autonomy_teacher import PickupTeacher
from autonomy_full_teacher import FullTeacher
from autonomy_receiver_fit import attach_receiver_head,fit,recovery_samples,sha,RECEIVER_RECIPE
from autonomy_recovery_mask import recovery_execution_mask
from autonomy_handover_refine import freeze_collection_model

ROOT=Path(__file__).resolve().parents[2]
POLICY=ROOT/'logs/grasp_v2/autonomy_motion_positions_v42/runtime/policy.pt'


class ReceiverChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        cls.device='cuda:0';torch.set_num_threads(4)

    def test_shadow_reference_sync_preserves_unselected_environment_and_physics(self):
        teacher=PickupTeacher.__new__(PickupTeacher)
        for key,shape in [('p',(2,2,3)),('q',(2,2,4)),('jaw',(2,2))]:setattr(teacher,key,torch.zeros(shape,device=self.device))
        teacher.stage=torch.tensor([0,5],device=self.device)
        ref=SimpleNamespace(p=torch.ones_like(teacher.p),q=torch.ones_like(teacher.q),jaw=torch.ones_like(teacher.jaw)*.2)
        before=copy.deepcopy(ref);stage=teacher.stage.clone()
        teacher.sync_executed_reference(ref,torch.tensor([True,False],device=self.device))
        self.assertTrue(torch.equal(stage,teacher.stage))
        for key in ('p','q','jaw'):
            torch.testing.assert_close(getattr(teacher,key)[0],getattr(ref,key)[0])
            self.assertEqual(float(getattr(teacher,key)[1].abs().sum()),0)
            torch.testing.assert_close(getattr(ref,key),getattr(before,key))

    def test_receiver_handoff_requires_real_lift_and_does_not_set_progress(self):
        teacher=FullTeacher.__new__(FullTeacher);d=self.device
        state=dict(tip=torch.randn(2,2,3,device=d),tool_q=torch.randn(2,2,4,device=d),
                   relative=torch.randn(2,2,3,device=d),relative_q=torch.randn(2,2,4,device=d))
        flags=torch.tensor([[True,True,False,False],[True,False,False,False]],device=d)
        teacher.env=SimpleNamespace(state=state,progress=SimpleNamespace(flags=flags))
        teacher.strong=lambda:torch.ones(2,2,device=d,dtype=torch.bool)
        ref=SimpleNamespace(p=state['tip'].clone(),q=state['tool_q'].clone(),jaw=torch.ones(2,2,device=d)*.15)
        for key in ('p','q','jaw','hold','holdq','relative','relative_q','receiver_anchor','center','dwell','receiver_age','hold_age','release_start'):
            source={'p':ref.p,'q':ref.q,'jaw':ref.jaw,'hold':ref.p,'holdq':ref.q,'relative':state['relative'],
                    'relative_q':state['relative_q'],'receiver_anchor':state['tip'][:,1],'center':state['tip'][:,1],
                    'dwell':flags[:,0].float(),'receiver_age':flags[:,0].float(),'hold_age':flags[:,0].float(),'release_start':ref.jaw}[key]
            setattr(teacher,key,torch.zeros_like(source))
        teacher.receiver_open=.23;teacher.stage=torch.zeros(2,device=d,dtype=torch.long)
        before=flags.clone();teacher.begin_receiver_recovery(ref,torch.tensor([True,False],device=d))
        self.assertEqual(teacher.stage.tolist(),[9,0]);self.assertTrue(torch.equal(flags,before))
        with self.assertRaises(ValueError):teacher.begin_receiver_recovery(ref,torch.tensor([False,True],device=d))

    def test_delayed_lift_and_exhausted_environments_cannot_trigger_invalid_handoff(self):
        d=self.device
        previous=torch.tensor([True,True,False,True],device=d)
        proposed=torch.zeros_like(previous)
        eligible=torch.tensor([True,True,True,False],device=d)
        ready=torch.tensor([False,True,False,True],device=d)
        actual=recovery_execution_mask(proposed,previous,eligible,ready)
        self.assertEqual(actual.tolist(),[True,False,False,False])
        handoff=previous&~actual&eligible
        self.assertEqual(handoff.tolist(),[False,True,False,False])
        ready[0]=True
        after=recovery_execution_mask(proposed,actual,eligible,ready)
        self.assertEqual((actual&~after&eligible).tolist(),[True,False,False,False])
        # Only collection paths change; parent weights and physics stay intact.
        parent=ROOT/'.checkpoint_runs/checkpoint11-assisted-full-bc-yaw5/model/policy.pt'
        with tempfile.TemporaryDirectory() as temporary:
            runtime=freeze_collection_model(parent,Path(temporary)/'runtime')
            self.assertEqual(sha(runtime),sha(parent))
            manifest=json.loads((runtime.parent/'manifest.json').read_text())
            original=json.loads((parent.parent/'manifest.json').read_text())
            changed={k for k,v in original['source_hashes'].items() if manifest['source_hashes'][k]!=v}
            self.assertEqual(changed,{'autonomy_full_teacher.py','autonomy_main.py'})
            source=(runtime.parent/'sources/autonomy_main.py').read_text()
            compile(source,'collection_main','exec')
            self.assertIn("expert_execution_mask_recorded=('expert_executed' in data)",source)

    def test_interrupted_collection_reuses_only_measured_complete_trajectories(self):
        dataset=ROOT/'logs/grasp_v2/autonomy_handover_refine_v55/recoveries/completed_demonstrations.npz'
        data,meta=recovery_samples(dataset,self.device,allow_completed_receipt=True)
        self.assertFalse(meta['collection_complete'])
        self.assertEqual(len(data['episode'][~data['validation']].unique()),10)
        self.assertEqual(data['episode'][data['validation']].unique().tolist(),[0,1])
        from autonomy_receiver_fit import completed_trajectory_receipt
        bad=copy.deepcopy(meta);bad['resets']['0']['yaw_deg']+=1
        with self.assertRaises(ValueError):completed_trajectory_receipt(dataset,bad)
        with self.assertRaises(FileNotFoundError):recovery_samples(dataset,self.device)

    def test_receiver_head_save_load_and_training_cannot_change_preserved_motors(self):
        d=self.device;parent=load_policy(POLICY,d);net=load_policy(POLICY,d)
        event=json.loads((ROOT/'logs/grasp_v2/autonomy_handover_diagnostic_v43/policy_events.jsonl').read_text().splitlines()[0])
        x=torch.tensor(event['physical_observations'][0],device=d)[None].repeat(19,1)
        options=torch.arange(19,device=d)
        attach_receiver_head(net)
        with torch.no_grad():
            net.receiver_jaw_rate_head.net[-1].bias.fill_(-10);net.receiver_jaw_rate_head.net[-1].bias[0]=10
        old=parent.jaw_rates(x,options);new=net.jaw_rates(x,options)
        torch.testing.assert_close(old[:,0],new[:,0],rtol=0,atol=0)
        keep=(options!=10)&(options!=11)
        torch.testing.assert_close(old[keep,1],new[keep,1],rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);collection=folder/'collection';collection.mkdir()
            manifest=json.loads((POLICY.parent/'manifest.json').read_text())
            (collection/'manifest.json').write_text(json.dumps(manifest))
            obs=x[:1].cpu().numpy().repeat(400,axis=0)
            actions=np.zeros((400,14),dtype=np.float32)
            actions[:,13]=4*(obs[:,69]*.5-.0002)-1
            stages=np.tile(np.repeat([10,11],20),10);episodes=np.repeat(np.arange(10),40);validation=np.isin(episodes,[2,7])
            report=dict(complete=True,assisted_grasp=True,teacher_at_inference=True,cases=[dict(success=True,full_success=True,length=40)]*10)
            (collection/'evaluation.json').write_text(json.dumps(report))
            (collection/'environment_close.json').write_text(json.dumps(dict(environment_closed=True,evaluation_sha256=sha(collection/'evaluation.json'))))
            meta=dict(schema=SCHEMA,skill='full',collection_complete=True,assisted_grasp=True,episodes=list(range(10)),parameters=dict(
                receiver_recovery_handoff=True,sync_learner_reference=True,learner_prefix_steps=2,seed=42,**RECEIVER_RECIPE))
            dataset=collection/'demonstrations.npz'
            np.savez(dataset,obs=obs,actions=actions,stage=stages,episode=episodes,validation=validation,metadata=json.dumps(meta))
            data,_=recovery_samples(dataset,d)
            self.assertEqual(len(data['obs']),380)
            # A delayed handoff must discard shadow labels even after the
            # nominal prefix; non-continuous ownership is rejected.
            expert=np.tile(np.arange(40)>=8,10)
            meta['expert_execution_mask_recorded']=True
            np.savez(dataset,obs=obs,actions=actions,stage=stages,episode=episodes,validation=validation,
                     expert_executed=expert,metadata=json.dumps(meta))
            delayed,_=recovery_samples(dataset,d)
            self.assertEqual(len(delayed['obs']),320)
            expert[0]=True
            np.savez(dataset,obs=obs,actions=actions,stage=stages,episode=episodes,validation=validation,
                     expert_executed=expert,metadata=json.dumps(meta))
            with self.assertRaises(ValueError):recovery_samples(dataset,d)
            meta.pop('expert_execution_mask_recorded')
            np.savez(dataset,obs=obs,actions=actions,stage=stages,episode=episodes,validation=validation,metadata=json.dumps(meta))
            outcome=fit(POLICY,dataset,folder/'candidate',d,updates=2,batch_size=64)
            self.assertTrue(outcome['physical_evaluation_required']);self.assertFalse(outcome['native_full_policy_verified'])
            result=load_policy(folder/'candidate/policy.pt',d)
            for key,value in parent.state_dict().items():torch.testing.assert_close(result.state_dict()[key],value,rtol=0,atol=0)
            with torch.inference_mode():
                torch.testing.assert_close(parent.jaw_rates(x,options)[:,0],result.jaw_rates(x,options)[:,0],rtol=0,atol=0)
            # Refining an already trained receiver must protect every other
            # tensor and its donor/non-contact outputs, including save/load.
            refined_parent=folder/'candidate/policy.pt'
            (collection/'manifest.json').write_text((refined_parent.parent/'manifest.json').read_text())
            refined=fit(refined_parent,dataset,folder/'refined',d,updates=2,batch_size=64,
                        refine_existing=True,preserve_parent_sources=True,learning_rate=2e-5,regularization=.1)
            self.assertTrue(refined['refine_existing_receiver'])
            check=load_policy(folder/'refined/policy.pt',d)
            for key,value in result.state_dict().items():
                if not key.startswith('receiver_jaw_rate_head.'):
                    torch.testing.assert_close(check.state_dict()[key],value,rtol=0,atol=0)
            with torch.inference_mode():
                a=result.jaw_rates(x,options);b=check.jaw_rates(x,options)
                torch.testing.assert_close(a[:,0],b[:,0],rtol=0,atol=0)
                torch.testing.assert_close(a[keep,1],b[keep,1],rtol=0,atol=0)
            self.assertEqual(json.loads((folder/'refined/manifest.json').read_text())['source_hashes'],
                             json.loads((refined_parent.parent/'manifest.json').read_text())['source_hashes'])
            # Whole-episode leakage must be rejected even for successful rows.
            validation[0]=True
            np.savez(dataset,obs=obs,actions=actions,stage=stages,episode=episodes,validation=validation,metadata=json.dumps(meta))
            with self.assertRaises(ValueError):recovery_samples(dataset,d)
            validation[0]=False;meta['parameters']['receiver_wall_angle']=60.
            np.savez(dataset,obs=obs,actions=actions,stage=stages,episode=episodes,validation=validation,metadata=json.dumps(meta))
            with self.assertRaises(ValueError):recovery_samples(dataset,d)


if __name__=='__main__':unittest.main(verbosity=2)
