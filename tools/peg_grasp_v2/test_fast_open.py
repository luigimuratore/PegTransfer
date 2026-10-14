"""CUDA tests for learned opening speed, protected closing and contact."""
import unittest
import torch
from test_transit_motion import TransitTests
from autonomy_fast_open import FastOpenMotorReference
from autonomy_fast_transit import FastTransitMotorReference


class OpenTests(TransitTests):
    def test_empty_opening_takes_up_shorter_approach_without_inventing_commands(self):
        s=self.state();old=FastTransitMotorReference(s,self.profile,4,16)
        new=FastOpenMotorReference(s,self.profile,4,16)
        for _ in range(60):
            a=torch.zeros(2,2,7,device=self.device)
            a[...,6]=4*(old.jaw+.0015)-1
            old.advance(a.flatten(1),s)
            a[...,6]=4*(new.jaw+.0015)-1
            new.advance(a.flatten(1),s)
        self.assertAlmostEqual(float(old.jaw[0,1]),.24,places=5)
        self.assertGreater(float(new.jaw[0,1]),.30)
        self.assertTrue(torch.equal(old.jaw[:,0],new.jaw[:,0])) # Near peg.
        before=new.jaw.clone();a[...,6]=4*before-1
        new.advance(a.flatten(1),s)
        self.assertTrue(torch.allclose(new.jaw,before,atol=1e-7))

    def test_closing_and_contact_opening_remain_identical(self):
        for contact,opening in ((False,False),(True,False),(True,True)):
            s=self.state();s['tip'][:,0]=s['tip'][:,1]
            if contact:self.grip(s)
            old=FastTransitMotorReference(s,self.profile,4,16)
            new=FastOpenMotorReference(s,self.profile,4,16)
            for _ in range(50):
                a=torch.zeros(2,2,7,device=self.device)
                a[...,6]=4*(old.jaw+(.0015 if opening else -.0002))-1
                old.advance(a.flatten(1),s);new.advance(a.flatten(1),s)
                self.assertTrue(torch.equal(old.jaw[:,0],new.jaw[:,0]))
                self.assertTrue(torch.equal(old.p,new.p));self.assertTrue(torch.equal(old.q,new.q))

    def test_no_physical_writes_and_no_extra_opening_above_boost_band(self):
        s=self.state();original={k:v.clone() for k,v in s.items()}
        old=FastTransitMotorReference(s,self.profile,4,16);new=FastOpenMotorReference(s,self.profile,4,16)
        old.jaw.fill_(.35);new.jaw.fill_(.35)
        a=torch.zeros(2,2,7,device=self.device);a[...,6]=4*(old.jaw+.0015)-1
        old.advance(a.flatten(1),s);new.advance(a.flatten(1),s)
        self.assertTrue(torch.equal(old.jaw,new.jaw))
        for key,value in original.items():self.assertTrue(torch.equal(value,s[key]),key)


if __name__=='__main__':unittest.main(defaultTest='OpenTests',verbosity=2)
