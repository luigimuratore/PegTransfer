"""CUDA checks: fast travel cannot bypass slow grip, handover or insertion."""
import unittest
import torch
from test_transit_motion import TransitTests
from autonomy_fast_transit import FastTransitMotorReference
from autonomy_env import ClearanceMotorReference


class FastTests(TransitTests):
    def test_high_caps_preserve_measured_interaction_boundaries(self):
        s=self.state();self.grip(s);s['bottom'].fill_(.05)
        ref=FastTransitMotorReference(s,self.profile,12,32)
        speed,angular,lead,_=self.tick(ref,s,400)
        self.assertAlmostEqual(float(speed[0,0]),.096,places=6)
        self.assertAlmostEqual(float(speed[0,1]),.3,places=6)
        self.assertAlmostEqual(float(angular[0,0]),.2,places=6)
        self.assertAlmostEqual(float(angular[0,1]),3.,places=6)
        self.assertAlmostEqual(float(lead[0,0]),.001,places=6)
        for fault in ('handover','insertion','slip','force','missing_contact'):
            t={k:v.clone() for k,v in s.items()}
            if fault=='handover':t['tip'][:,1]=t['tip'][:,0]
            elif fault=='insertion':t['bottom'].fill_(.042)
            elif fault=='slip':t['relative'][:,0,0]+=.002
            elif fault=='force':t['forces'][:,0,0,0]=.8
            else:t['contact_valid'][:,0]=False
            speed,angular,_,_=self.tick(ref,t,1)
            self.assertAlmostEqual(float(speed[0,0]),.003,places=6,msg=fault)
            self.assertAlmostEqual(float(angular[0,0]),.2,places=6,msg=fault)
        s=self.state();ref=FastTransitMotorReference(s,self.profile,12,32)
        _,angular,_,_=self.tick(ref,s,400)
        self.assertAlmostEqual(float(angular[0,0]),1.,places=6)

    def test_old_caps_are_identical_in_new_controller(self):
        s=self.state(8);self.grip(s)
        old=ClearanceMotorReference(s,self.profile,4,4,True)
        new=FastTransitMotorReference(s,self.profile,4,4)
        rng=torch.Generator(device=self.device).manual_seed(6161)
        for step in range(100):
            a=torch.rand((8,14),device=self.device,generator=rng)*2-1
            if step==50:s['forces'].zero_()
            old.advance(a,s);new.advance(a,s)
            for key in ('p','q','jaw','speed_cap','grip_age'):
                self.assertTrue(torch.equal(getattr(old,key),getattr(new,key)),key)
            s['tip']=old.p.clone();s['tool_q']=old.q.clone()

    def test_high_caps_leave_jaws_and_physical_state_unchanged(self):
        s=self.state();self.grip(s);original={k:v.clone() for k,v in s.items()}
        old=ClearanceMotorReference(s,self.profile,2,2,True)
        new=FastTransitMotorReference(s,self.profile,12,32)
        a=torch.ones(2,14,device=self.device);a[:,6]=-1
        for _ in range(50):
            old.advance(a,s);new.advance(a,s)
            self.assertTrue(torch.equal(old.jaw,new.jaw))
        for k,v in original.items():self.assertTrue(torch.equal(v,s[k]),k)


if __name__=='__main__':
    unittest.main(defaultTest='FastTests',verbosity=2)
