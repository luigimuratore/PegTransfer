"""Measured two-arm transition checks; no simulation or training."""
import copy
import json
import unittest
import numpy as np
from types import SimpleNamespace
from full_task import secure_contact,fixture_holds
from full_geometry import placement_evidence,screen_full_route,TARGET_XY
from geometry import load_profile
from replay import checkpoint_sources,REPO,FRONT_EYE
from handover_grip import WallReceiverGrip


class PhysicalTransferTests(unittest.TestCase):
    def test_only_active_unbroken_fixed_support_can_replace_pressure_hold(self):
        fixed=SimpleNamespace(holder=1,break_events=[],evidence=lambda:dict(type='temporary_physx_fixed_joint'))
        self.assertTrue(fixture_holds(fixed,1));self.assertFalse(fixture_holds(fixed,2))
        fixed.break_events.append('broken')
        self.assertFalse(fixture_holds(fixed,1))
        spring=SimpleNamespace(holder=1,break_events=[],evidence=lambda:dict(type='bounded_virtual_spring'))
        self.assertFalse(fixture_holds(spring,1));self.assertFalse(fixture_holds(None,1))

    def test_receiver_unloads_first_contact_before_building_pressure(self):
        grip=WallReceiverGrip('wall',.15)
        row=dict(finger_normal_force_n=[.01,0],finger_force_vectors_n=[[.01,0,0],[0,0,0]],
                 finger_contact_points_w_m=[[0,0,0],None],finger_contact_surfaces=['outer','none'],
                 tool_tip_w_m=[0,0,0],tool_quaternion_wxyz=[1,0,0,0],peg_in_tool_m=[.005,0,0],
                 peg_quat_in_tool=[1,0,0,0])
        first=grip.update(row,'close',.01)
        self.assertTrue(first['jaw_closure_paused']);self.assertEqual(first['jaw_target_rad'],.15)
        self.assertLess(first['centering_offset_mm'],0.);self.assertFalse(first['pressure_ready'])
        row.update(finger_normal_force_n=[.35,.35],finger_force_vectors_n=[[.35,0,0],[-.35,0,0]],
                   finger_contact_points_w_m=[[0,0,0],[.002,0,0]],finger_contact_surfaces=['outer','inner'])
        for _ in range(40):ready=grip.update(row,'close',.01)
        self.assertTrue(ready['pressure_ready']);self.assertIsNotNone(grip.reference)

    def test_receiver_needs_both_opposing_wall_contacts(self):
        row=dict(finger_normal_force_n=[.35,.35],finger_force_vectors_n=[[.35,0,0],[-.35,0,0]],
                 finger_contact_points_w_m=[[0,0,0],[.002,0,0]],finger_contact_surfaces=['inner','outer'])
        self.assertTrue(secure_contact(row,.35))
        for key,value in [('finger_normal_force_n',[.35,0]),('finger_force_vectors_n',[[.35,0,0],[.35,0,0]]),
                          ('finger_contact_surfaces',['outer','outer'])]:
            wrong=copy.deepcopy(row);wrong[key]=value;self.assertFalse(secure_contact(wrong,.35))

    def test_placement_requires_correct_hole_release_and_rest(self):
        profile=load_profile();root=[TARGET_XY[0]-profile['hole_center_m'][0],TARGET_XY[1],.024613]
        row=dict(peg_root_w_m=root,peg_quaternion_wxyz=[1,0,0,0],peg_bottom_m=.010,
                 peg_speed_m_s=0.,finger_normal_force_n=[0,0])
        self.assertTrue(placement_evidence(row,profile)['placed'])
        for key,value in [('peg_root_w_m',[root[0]+.003,*root[1:]]),('peg_bottom_m',.020),
                          ('peg_speed_m_s',.02),('finger_normal_force_n',[.3,.3]),
                          ('peg_quaternion_wxyz',[2**-.5,2**-.5,0,0])]:
            wrong=copy.deepcopy(row);wrong[key]=value;self.assertFalse(placement_evidence(wrong,profile)['placed'])
            json.dumps(placement_evidence(wrong,profile),allow_nan=False)

    def test_full_route_keeps_joint_limits(self):
        result=screen_full_route(load_profile())
        self.assertTrue(result['reachable'],result);self.assertEqual(len(result['routes']),2)

    def test_checkpoint_sources_original_and_camera_opposite(self):
        folder,meta=checkpoint_sources(REPO/'logs/grasp_v2/wall_bc3/policy.npz')
        self.assertEqual(meta['schema'],'peg-wall-motor-bc-3')
        self.assertTrue((folder/'runtime.py').exists());self.assertGreater(FRONT_EYE[1],0)


if __name__=='__main__':unittest.main(verbosity=2)
