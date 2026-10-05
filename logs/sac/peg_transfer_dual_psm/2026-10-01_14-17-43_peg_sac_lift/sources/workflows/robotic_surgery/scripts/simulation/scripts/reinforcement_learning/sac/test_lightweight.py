"""CPU regression tests; never import or launch Isaac Sim."""

import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace as NS
import unittest

import gymnasium as gym
import numpy as np
import torch

from common import SCHEMA, TASK, TASK_DIR, fingerprint, read_checkpoint
from vec_env import PegSACVecEnv

# Import task-only torch code without executing its Isaac-dependent package __init__.
package = ModuleType('_peg_cpu')
package.__path__ = [str(TASK_DIR)]
sys.modules[package.__name__] = package
g = importlib.import_module('_peg_cpu.sac_geometry')
m = importlib.import_module('_peg_cpu.sac_mdp')


class Object:
    def __init__(self, n):
        pose = torch.zeros(n, 7)
        pose[:, 3] = 1
        self.data = NS(root_pose_w=pose, root_pos_w=pose[:, :3], root_quat_w=pose[:, 3:],
                       default_root_state=torch.cat((pose, torch.zeros(n, 6)), dim=1),
                       root_lin_vel_w=torch.zeros(n, 3), root_ang_vel_w=torch.zeros(n, 3))

    def write_root_pose_to_sim(self, pose, env_ids=None):
        self.data.root_pose_w[env_ids if env_ids is not None else slice(None)] = pose

    def write_root_velocity_to_sim(self, velocity, env_ids=None):
        ids = env_ids if env_ids is not None else slice(None)
        self.data.root_lin_vel_w[ids], self.data.root_ang_vel_w[ids] = velocity[:, :3], velocity[:, 3:]


def fake(n=2):
    scene = {'object': Object(n)}
    class Scene(dict):
        pass
    scene = Scene(scene)
    scene.env_origins = torch.tensor([[i * 2.5, 0., i * 0.5] for i in range(n)])
    for a in (1, 2):
        scene[f'robot_{a}'] = NS(data=NS(joint_pos=torch.ones(n, 2) * 0.5), find_joints=lambda _: ([0, 1], []))
        scene[f'ee_{a}_frame'] = NS(data=NS(target_pos_w=torch.zeros(n, 1, 3)))
    env = NS(scene=scene, num_envs=n, device='cpu', step_dt=0.02, common_step_counter=0,
             cfg=NS(source_post='L5', phase='full'), action_manager=NS(action=torch.zeros(n, 14)))
    m.reset(env)
    return env


def set_peg(env, p):
    env.scene['object'].data.root_pos_w[:] = torch.tensor(p) + env.scene.env_origins


def tick(env, count=1):
    for _ in range(count):
        env.common_step_counter += 1
        m.update(env)


class GeometryTests(unittest.TestCase):
    def test_compound_prisms_preserve_hole(self):
        from scipy.spatial import ConvexHull
        sectors = g.collider_sectors()
        self.assertEqual(len(sectors), 36)
        for sector in sectors:
            hull = ConvexHull(np.array(sector))
            # A 2 mm cylindrical post must fit at every z through every sector.
            for angle in np.linspace(0, 2 * np.pi, 80):
                probe = np.array([g.HOLE_CENTER[0] + g.POST_RADIUS * np.cos(angle),
                                  g.POST_RADIUS * np.sin(angle), -0.006])
                self.assertFalse(np.all(hull.equations[:, :3] @ probe + hull.equations[:, 3] <= 1e-10))
            self.assertGreater(hull.volume, 0)

    def test_vertical_exit_and_target_insertion(self):
        start = torch.tensor([[-g.HOLE_CENTER[0], 0., g.RESET_Z]])
        self.assertTrue(g.upright_path_safe(start, start).item())
        self.assertFalse(g.upright_path_safe(start, start + torch.tensor([[0.004, 0., 0.]])).item())
        while start[0, 2] < g.CLEAR_Z:
            end = start + torch.tensor([[0., 0., 0.004]])
            self.assertTrue(g.upright_path_safe(start, end).item())
            start = end
        self.assertTrue(g.upright_path_safe(start, start + torch.tensor([[0.004, 0., 0.]])).item())
        target = torch.tensor([g.TARGET])
        self.assertTrue(g.upright_path_safe(target + torch.tensor([[0., 0., .004]]), target).item())
        bad = target + torch.tensor([[.003, 0., 0.]])
        self.assertFalse(g.upright_path_safe(bad + torch.tensor([[0., 0., .004]]), bad).item())

    def test_all_post_axes_and_board(self):
        for x, y in g.POSTS:
            p = torch.tensor([[x - g.HOLE_CENTER[0], y, g.RESET_Z]])
            self.assertTrue(g.upright_path_safe(p, p).item())
            self.assertFalse(g.upright_path_safe(p, p + torch.tensor([[0., .004, 0.]])).item())
        p = torch.tensor([g.TARGET])
        self.assertFalse(g.upright_path_safe(p, p - torch.tensor([[0., 0., .001]])).item())
        slop = p - torch.tensor([[0., 0., .00005]])
        self.assertTrue(g.upright_path_safe(slop, p).item())


class SequenceTests(unittest.TestCase):
    def test_position_alone_is_not_success(self):
        env = fake()
        set_peg(env, g.TARGET)
        tick(env, 35)
        s = m.state(env)
        for k in m.METRICS:
            self.assertFalse(getattr(s, k).any(), k)

    def test_ordered_sequence_and_dwell(self):
        env = fake()
        s = m.state(env)
        set_peg(env, (-g.HOLE_CENTER[0], 0., .065))
        s.holder[:] = 1
        s.source_cleared[:] = True
        env.scene['robot_1'].data.joint_pos[:] = .07
        tick(env, 15)
        self.assertFalse(s.lift_stable.any())
        tick(env)
        self.assertTrue(s.lift_stable.all())
        s.receiver_latched[:] = True
        tick(env)
        self.assertTrue(s.grasp_psm2.all())
        s.holder[:] = 2
        s.receiver_latched[:] = False
        env.scene['robot_1'].data.joint_pos[:] = .5
        tick(env, 4)
        self.assertFalse(s.handover.any())
        tick(env)
        self.assertTrue(s.handover.all())
        set_peg(env, g.TARGET)
        s.holder[:] = 0
        for a in (1, 2):
            env.scene[f'ee_{a}_frame'].data.target_pos_w[:, 0] = env.scene.env_origins + torch.tensor([0., 0., .12])
        tick(env, 25)
        self.assertFalse(s.full_success.any())  # first placement step still moving
        tick(env)
        self.assertTrue(s.full_success.all())
        self.assertTrue(s.placement.all())
        for k in m.METRICS:
            self.assertTrue(getattr(s, k).all(), k)

    def test_tilted_or_offset_placement_rejected(self):
        env = fake()
        set_peg(env, g.TARGET)
        self.assertTrue(m.hole_on_target(env).all())
        env.scene['object'].data.root_pos_w[:, 0] += .001
        q = env.scene['object'].data.root_quat_w
        q[:, 0] = np.cos(.035)
        q[:, 2] = -np.sin(.035)
        self.assertTrue(m.upright(env).all())
        self.assertFalse(m.hole_on_target(env).any())

    def test_reset_preserves_other_environments(self):
        env = fake()
        s = m.state(env)
        s.full_success[:] = True
        s.holder[:] = 2
        s.highwater[:] = 1
        m.reset(env, torch.tensor([0]))
        self.assertFalse(s.full_success[0])
        self.assertTrue(s.full_success[1])
        self.assertEqual(s.holder.tolist(), [0, 2])
        self.assertEqual(s.highwater.sum(1).tolist(), [0, 6])

    def test_receiver_cannot_grasp_before_stable_lift(self):
        env = fake()
        s = m.state(env)
        set_peg(env, (-g.HOLE_CENTER[0], 0., .065))
        s.holder[:] = 1
        s.source_cleared[:] = True
        p = m.local_peg(env)
        env.scene['robot_1'].data.joint_pos[:] = .07
        env.scene['robot_2'].data.joint_pos[:] = .07
        for a, offset in ((1, g.GRASP1), (2, g.GRASP2)):
            env.scene[f'ee_{a}_frame'].data.target_pos_w[:, 0] = p + p.new_tensor(offset) + env.scene.env_origins
        s.offset[:] = -p.new_tensor(g.GRASP1)
        for _ in range(10):
            m.assist(env)
        self.assertFalse(s.receiver_latched.any())
        s.lift_stable[:] = True
        for _ in range(8):
            m.assist(env)
        self.assertTrue(s.receiver_latched.all())
        s.grasp_psm2[:] = True
        env.scene['ee_2_frame'].data.target_pos_w[:, :, 0] += .02
        env.scene['robot_1'].data.joint_pos[:] = .5
        m.assist(env)
        self.assertFalse(s.receiver_latched.any())
        self.assertEqual(s.holder.tolist(), [0, 0])

    def test_progress_cannot_be_farmed_by_oscillation(self):
        env = fake()
        total = torch.zeros(2)
        for i in range(100):
            p = m.local_peg(env)
            env.scene['ee_1_frame'].data.target_pos_w[:, 0] = p + p.new_tensor(g.GRASP1) + env.scene.env_origins
            if i % 2:
                env.scene['ee_1_frame'].data.target_pos_w[:, :, 0] += .1
            env.common_step_counter += 1
            total += m.reward(env) * env.step_dt
        self.assertTrue((total <= 0.001).all(), total)  # budget 2, elapsed cost 100 * 0.02


class FakeAutoReset:
    def __init__(self):
        self.unwrapped = self
        self.num_envs, self.device, self.render_mode = 2, 'cpu', None
        self.single_observation_space = {'policy': gym.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)}
        self.single_action_space = gym.spaces.Box(-np.inf, np.inf, (14,), dtype=np.float32)

    def reset(self):
        return {'policy': torch.zeros(2, 3)}, {}

    def step(self, actions):
        self._sac_terminal = {i: {'observation': np.full(3, 11 + i, np.float32),
                                  'metrics': {k: False for k in m.METRICS}, 'source': 'L5',
                                  'blocked_moves': 0, 'failure': i == 0, 'phase_success': False}
                              for i in range(2)}
        return {'policy': torch.full((2, 3), 999.)}, torch.ones(2), torch.tensor([True, False]), torch.tensor([False, True]), {}

    def close(self):
        pass


class ReplayTests(unittest.TestCase):
    def test_sac_updates_and_uses_true_terminal_observation(self):
        from stable_baselines3 import SAC
        env = PegSACVecEnv(FakeAutoReset())
        env.reset()
        _, _, dones, infos = env.step(np.zeros((2, 14)))
        np.testing.assert_array_equal(infos[1]['terminal_observation'], [12, 12, 12])
        self.assertTrue(infos[1]['TimeLimit.truncated'])
        self.assertFalse(infos[0]['TimeLimit.truncated'])
        settings = dict(importlib.import_module('_peg_cpu.agents.sac_cfg').SAC_SETTINGS)
        settings.update(buffer_size=64, batch_size=8, learning_starts=0, gradient_steps=1,
                        policy_kwargs={'net_arch': [16, 16], 'n_critics': 2}, verbose=0)
        model = SAC('MlpPolicy', env, device='cpu', seed=42, **settings)
        model.learn(8)
        self.assertGreater(model._n_updates, 0)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.policy.parameters()))
        np.testing.assert_array_equal(model.replay_buffer.next_observations[0], [[11]*3, [12]*3])
        np.testing.assert_array_equal(model.replay_buffer.timeouts[0], [0, 1])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'sac.zip'
            model.save(p)
            loaded = SAC.load(p, env=env, device='cpu')
            action, _ = loaded.predict(np.zeros((2, 3), dtype=np.float32), deterministic=True)
            self.assertEqual(action.shape, (2, 14))
            self.assertTrue((np.abs(action) <= 1).all())
            p.with_suffix('.json').write_text(json.dumps(dict(algorithm='SAC', task=TASK, schema=SCHEMA,
                                                             fingerprint=fingerprint())))
            self.assertEqual(read_checkpoint(p)['algorithm'], 'SAC')
            model.save_replay_buffer(Path(tmp) / 'replay.pkl')
            loaded.load_replay_buffer(Path(tmp) / 'replay.pkl')
            self.assertEqual(loaded.replay_buffer.pos, model.replay_buffer.pos)

    def test_reject_ppo_checkpoint_before_sim(self):
        with self.assertRaisesRegex(ValueError, 'PPO'):
            read_checkpoint(Path('model_1798.pt'))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
