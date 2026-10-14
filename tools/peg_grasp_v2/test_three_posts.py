"""CUDA checks of exact pair allocation and explicit force assistance."""
import unittest
from types import SimpleNamespace
import torch
from autonomy_three_posts import pair_indices
from autonomy_gripper_guard import GripperLoadGuard
from autonomy_upright_support import upright_torque
from rl_core import quaternion

class Checks(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():raise RuntimeError('RTX/CUDA required')
        self.device='cuda:0'
    def test_pair_sequence_has_no_random_replacement_or_duplicate(self):
        pairs,cursor=pair_indices(0,3,3,self.device)
        self.assertEqual(pairs.tolist(),[0,1,2])
        serial=[];cursor=0
        for _ in range(3):
            pair,cursor=pair_indices(cursor,1,3,self.device);serial+=pair.tolist()
        self.assertEqual(serial,[0,1,2])
    def test_guard_moves_only_overloaded_jaws_and_keeps_physical_inputs(self):
        guard=GripperLoadGuard(2,self.device)
        action=torch.randn(2,14,device=self.device).clamp(-1,1)
        action[:,6]=-1;action[:,13]=-1
        force=torch.zeros(2,2,2,3,device=self.device)
        force[0,1,0,0]=.8
        jaw=torch.full((2,2),.2,device=self.device)
        original_force=force.clone();original_jaw=jaw.clone();original_action=action.clone()
        result=guard.apply(action,{'forces':force},SimpleNamespace(jaw=jaw))
        keep=torch.ones_like(action,dtype=torch.bool);keep[0,13]=False
        torch.testing.assert_close(result[keep],action[keep],rtol=0,atol=0)
        self.assertGreater(float(result[0,13]),float(action[0,13]))
        for old,new in ((original_force,force),(original_jaw,jaw),(original_action,action)):
            torch.testing.assert_close(old,new,rtol=0,atol=0)
        self.assertFalse(guard.evidence()['task_phase_input'])
    def test_guard_does_not_slow_a_neural_release(self):
        guard=GripperLoadGuard(1,self.device)
        force=torch.ones(1,2,2,3,device=self.device)
        action=torch.zeros(1,14,device=self.device);action[:,[6,13]]=1
        result=guard.apply(action,{'forces':force},SimpleNamespace(jaw=torch.full((1,2),.2,device=self.device)))
        torch.testing.assert_close(result,action,rtol=0,atol=0)
    def test_upright_support_is_bounded_and_keeps_yaw_and_free_object(self):
        vectors=torch.tensor([[0.,0.,.7],[.2,0.,0.],[.2,0.,0.]],device=self.device)
        q=quaternion(vectors);angular=torch.zeros(3,3,device=self.device)
        moment=upright_torque(q,angular,torch.tensor([True,True,False],device=self.device))
        self.assertLess(float(moment[0].norm()),1e-6)
        self.assertLess(float(moment[1,0]),0)
        self.assertLessEqual(float(moment.norm(dim=-1).max()),.0040001)
        self.assertEqual(float(moment[2].norm()),0)

if __name__=='__main__':unittest.main(verbosity=2)
