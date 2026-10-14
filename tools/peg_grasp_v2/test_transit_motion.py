"""CUDA regressions for frozen default motion and measured transit caps."""
import importlib.util
import json
from pathlib import Path
import unittest
import torch
from autonomy_env import ContactMotorReference,ClearanceMotorReference
from rl_core import quaternion

ROOT=Path(__file__).resolve().parents[2]


class TransitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        cls.device='cuda:0';torch.set_num_threads(4)
        cls.profile=json.loads((ROOT/'tools/peg_grasp_v2/peg_profile.json').read_text())
    def state(self,n=2):
        d=self.device
        tip=torch.tensor([[[.009,0.,.055],[.080,.080,.080]]]*n,device=d)
        peg=torch.tensor([[0.,0.,.05]]*n,device=d)
        q=quaternion(torch.zeros(n,2,3,device=d))
        return dict(tip=tip,peg=peg,tool_q=q,peg_q=q[:,0].clone(),
            jaws=torch.full((n,2,2),.15,device=d),forces=torch.zeros(n,2,2,3,device=d),
            contact_valid=torch.zeros(n,2,2,dtype=torch.bool,device=d),
            relative=peg[:,None]-tip,relative_q=q.clone(),bottom=torch.full((n,),.045,device=d))
    def grip(self,s):
        s['forces'][:,0,0,0]=.3;s['forces'][:,0,1,0]=-.3;s['contact_valid'][:,0]=True
    def tick(self,ref,s,steps=100):
        contact=s['forces'].norm(dim=-1).amax(-1)>.002
        for _ in range(steps):limits=ref.limits(s,contact)
        return limits
    def test_original_profile_is_bit_identical_to_preserved_v48(self):
        path=ROOT/'logs/grasp_v2/autonomy_receiver_bc_v48/receiver_bc/sources/autonomy_env.py'
        spec=importlib.util.spec_from_file_location('v48_original_motor',path)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        s=self.state(32);before=old.ContactMotorReference(s,1.5,1.25);after=ContactMotorReference(s,1.5,1.25)
        rng=torch.Generator(device=self.device).manual_seed(707)
        for step in range(100):
            s['forces']=torch.rand(s['forces'].shape,generator=rng,device=self.device)*(.1 if step%2 else 0.)
            a=torch.rand((32,14),generator=rng,device=self.device)*2-1
            before.advance(a,s);after.advance(a,s)
            for key in ('p','q','jaw'):self.assertTrue(torch.equal(getattr(before,key),getattr(after,key)),key)
            s['tip']=before.p.clone();s['tool_q']=before.q.clone()
    def test_stable_loaded_transport_accelerates_but_handover_and_insertion_slow(self):
        s=self.state();self.grip(s);ref=ClearanceMotorReference(s,self.profile,3.,4.)
        speed,angular,lead,_=self.tick(ref,s)
        self.assertAlmostEqual(float(speed[0,0]),.012,places=6)
        self.assertAlmostEqual(float(angular[0,0]),.2,places=6)
        self.assertAlmostEqual(float(lead[0,0]),.001,places=6)
        # Receiver approaching object: donor transport slows immediately.
        s['tip'][:,1]=s['tip'][:,0]
        speed,_,_,_=self.tick(ref,s,1)
        self.assertAlmostEqual(float(speed[0,0]),.003,places=6)
        s['tip'][:,1]=torch.tensor([.08,.08,.08],device=self.device)
        s['bottom'].fill_(.029)
        speed,_,_,_=self.tick(ref,s,1)
        self.assertAlmostEqual(float(speed[0,0]),.003,places=6)
    def test_unopposed_or_slipping_grip_cannot_enable_fast_transport(self):
        for fault in ('opposition','slip','pressure','contact_valid'):
            s=self.state();self.grip(s);ref=ClearanceMotorReference(s,self.profile,2.,4.)
            self.tick(ref,s)
            if fault=='opposition':s['forces'][:,0,1,0]=.3
            elif fault=='slip':s['relative'][:,0,0]+=.002
            elif fault=='pressure':s['forces'][:,0,0,0]=.8
            else:s['contact_valid'][:,0,1]=False
            speed,_,_,_=self.tick(ref,s,1)
            self.assertAlmostEqual(float(speed[0,0]),.003,places=6,msg=fault)
    def test_free_motion_ramps_and_slows_near_peg_or_board_without_contact(self):
        s=self.state();ref=ClearanceMotorReference(s,self.profile,4.,4.)
        speed,_,_,_=self.tick(ref,s,1)
        self.assertLessEqual(float(speed[0,1]),.004001)
        speed,_,_,_=self.tick(ref,s,120)
        self.assertAlmostEqual(float(speed[0,1]),.1,places=6)
        self.assertAlmostEqual(float(speed[0,0]),.005,places=6)
        s['tip'][:,1,2]=.029
        speed,_,_,_=self.tick(ref,s,1)
        self.assertAlmostEqual(float(speed[0,1]),.005,places=6)
    def test_reset_does_not_erase_other_environment_grip_history(self):
        s=self.state();self.grip(s);ref=ClearanceMotorReference(s,self.profile,2.,3.)
        self.tick(ref,s);other=ref.grip_age[1].clone();other_speed=ref.speed_cap[1].clone()
        ref.reset(s,torch.tensor([0],device=self.device))
        self.assertEqual(float(ref.grip_age[0].sum()),0.)
        self.assertTrue(torch.equal(other,ref.grip_age[1]));self.assertTrue(torch.equal(other_speed,ref.speed_cap[1]))
    def test_transit_caps_do_not_change_jaw_command_or_write_physical_state(self):
        s=self.state();self.grip(s);original={k:v.clone() for k,v in s.items()}
        old=ContactMotorReference(s);new=ClearanceMotorReference(s,self.profile,4.,4.)
        a=torch.tensor([[1.,0.,0.,0.,0.,0.,-1.]*2]*2,device=self.device)
        for _ in range(50):
            old.advance(a,s);new.advance(a,s)
            self.assertTrue(torch.equal(old.jaw,new.jaw))
        for key,value in original.items():self.assertTrue(torch.equal(value,s[key]),key)
    def test_travel_unit_scales_match_original_after_contact_loss_and_resets(self):
        s=self.state();old=ContactMotorReference(s)
        new=ClearanceMotorReference(s,self.profile,1.,1.,True)
        rng=torch.Generator(device=self.device).manual_seed(808)
        for step in range(100):
            s['forces'].zero_()
            if step%3:self.grip(s)
            a=torch.rand((2,14),device=self.device,generator=rng)*2-1
            old.advance(a,s);new.advance(a,s)
            for key in ('p','q','jaw'):self.assertTrue(torch.equal(getattr(old,key),getattr(new,key)),key)
            s['tip']=old.p.clone();s['tool_q']=old.q.clone()
            if step==50:
                ids=torch.tensor([0],device=self.device);old.reset(s,ids);new.reset(s,ids)
    def test_travel_preserves_rotation_and_close_approach_but_speeds_clear_translations(self):
        s=self.state();ref=ClearanceMotorReference(s,self.profile,4.,4.,True)
        speed,angular,_,_=self.tick(ref,s)
        self.assertAlmostEqual(float(speed[0,0]),.025,places=6)
        self.assertAlmostEqual(float(speed[0,1]),.1,places=6)
        self.assertTrue(torch.equal(angular,torch.ones_like(angular)))
        self.grip(s);s['bottom'].fill_(.042)
        speed,angular,_,_=self.tick(ref,s)
        self.assertAlmostEqual(float(speed[0,0]),.003,places=6)
        self.assertAlmostEqual(float(angular[0,0]),.2,places=6)
        s['bottom'].fill_(.050)
        speed,_,_,_=self.tick(ref,s)
        self.assertAlmostEqual(float(speed[0,0]),.012,places=6)


if __name__=='__main__':unittest.main(verbosity=2)
