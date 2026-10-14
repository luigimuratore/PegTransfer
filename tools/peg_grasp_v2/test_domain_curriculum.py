"""Check invalid resets and missing post-pair coverage before GPU experiments."""
import ast
import copy
import json
from pathlib import Path
import unittest
import torch
from autonomy_env import SOURCES, TARGETS, AutonomyEnv
from autonomy_domain_curriculum import curriculum_levels, pair_gate, level_gate, SEEDS
from geometry import load_profile, validate_reset_range
from rl_core import apply

ROOT = Path(__file__).resolve().parents[2]


class DomainChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        cls.profile = load_profile()

    def test_post_coordinates_match_preserved_physics(self):
        path = ROOT/'.checkpoint_runs/checkpoint3-full-assisted/workflows/robotic_surgery/scripts/simulation/exts/robotic.surgery.tasks/robotic/surgery/tasks/surgical/peg_transfer/scene_cfg.py'
        tree = ast.parse(path.read_text())
        found = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            if not name.startswith('post_'):
                continue
            for call in ast.walk(node.value):
                if isinstance(call, ast.Call):
                    for kw in call.keywords:
                        if kw.arg == 'pos':
                            found[name[5:].upper()] = tuple(ast.literal_eval(x) for x in kw.value.elts[:2])
        self.assertEqual(found, {**SOURCES, **TARGETS})

    def test_reset_overlap_is_rejected_at_all_yaws(self):
        for yaw in (0., 30., 180.):
            self.assertGreater(validate_reset_range(self.profile, 1.1, yaw), .0001)
            with self.assertRaises(ValueError):
                validate_reset_range(self.profile, 1.5, yaw)
        for jitter, yaw in ((float('nan'), 5.), (.5, float('nan')), (-.1, 5.), (.5, 181.)):
            with self.assertRaises(ValueError):
                validate_reset_range(self.profile, jitter, yaw)

    def test_wider_rotation_keeps_hole_on_selected_post_cuda(self):
        from types import SimpleNamespace
        dev = 'cuda:0'
        env = AutonomyEnv.__new__(AutonomyEnv)
        env.device = dev;env.profile = self.profile
        env.source_names = ['L2', 'L6'];env.target_names = ['R4', 'R6']
        env.source_values = torch.tensor([SOURCES[x] for x in env.source_names], device=dev)
        env.target_values = torch.tensor([TARGETS[x] for x in env.target_names], device=dev)
        env.reset_rng = torch.Generator(device=dev).manual_seed(42)
        env.jitter = .0011;env.yaw = torch.pi/6
        env.source = torch.zeros(32, 3, device=dev);env.source[:,2] = .010
        env.target = env.source.clone();env.origins = torch.zeros_like(env.source)
        env.reset_records = [None]*32
        poses = torch.zeros(32, 7, device=dev)
        obj = SimpleNamespace(data=SimpleNamespace(root_state_w=poses))
        obj.write_root_pose_to_sim = lambda pose, env_ids: poses.index_copy_(0, env_ids, pose)
        obj.write_root_velocity_to_sim = lambda *args, **kw: None
        class Scene(dict):
            def update(self, dt): pass
        env.env = SimpleNamespace(scene=Scene(object=obj))
        ids = torch.arange(32, device=dev)
        env.randomize(ids)
        hole = torch.tensor([*self.profile['hole_center_m'], 0.], device=dev).expand(32,-1)
        actual = poses[:,:3]+apply(poses[:,3:], hole)
        offsets = torch.tensor([r['jitter_mm'] for r in env.reset_records], device=dev)/1000
        torch.testing.assert_close(actual[:,:2], env.source[:,:2]+offsets, atol=1e-8, rtol=0)
        for r in env.reset_records:
            self.assertEqual(r['target_post_center_m'][:2], list(env.target_values[env.target_names.index(r['target'])].cpu().numpy()))

    def test_gate_rejects_wrong_post_or_speed_despite_full_success(self):
        report = json.loads((ROOT/'.checkpoint_runs/checkpoint10-assisted-full-bc-transit/reports/transit0_seed54321.json').read_text())
        for row in report['cases']:
            row['reset']['target_post_center_m'] = [*TARGETS['R2'], .01]
        self.assertTrue(pair_gate(report, 'L5', 'R2', .5, 2.))
        bad = copy.deepcopy(report);bad['cases'][0]['reset']['target_post_center_m'][0] += .042
        self.assertFalse(pair_gate(bad, 'L5', 'R2', .5, 2.))
        bad = copy.deepcopy(report);bad['motion_reference']['transport_scale'] = 3.
        self.assertFalse(pair_gate(bad, 'L5', 'R2', .5, 2.))

    def test_gate_requires_each_pair_and_seed(self):
        level = next(x for x in curriculum_levels() if x['name'] == 'two_sources')
        rows = [dict(source=s, target=t, seed=seed, passed=True, jitter_mm=.25, yaw_deg=2.)
                for s in level['sources'] for t in level['targets'] for seed in SEEDS]
        self.assertTrue(level_gate(rows, level))
        self.assertFalse(level_gate(rows[:-1], level))
        rows[-1]['passed'] = False
        self.assertFalse(level_gate(rows, level))


if __name__ == '__main__':
    unittest.main(verbosity=2)
