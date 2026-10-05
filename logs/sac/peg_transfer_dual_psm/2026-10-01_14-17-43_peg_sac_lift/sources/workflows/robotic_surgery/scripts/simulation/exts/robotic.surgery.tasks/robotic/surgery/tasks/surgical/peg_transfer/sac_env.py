"""Isaac Lab 0.48 ManagerBasedRLEnv with assistance inside the physics loop.

The step sequence follows the locally installed ManagerBasedRLEnv API. Terminal
observations and metrics are captured before asynchronous resets for off-policy RL.
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv

from . import sac_mdp as mdp


class PegTransferSACEnv(ManagerBasedRLEnv):
    def step(self, action):
        self._sac_terminal = {}
        self.action_manager.process_action(action.to(self.device).clamp(-1, 1))
        self.recorder_manager.record_pre_step()
        rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            mdp.assist(self)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and rendering:
                self.sim.render()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        mdp.update(self)
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()
        ids = self.reset_buf.nonzero().flatten()
        if len(ids):
            # Reward highwater state has been updated, so this is the true next observation.
            terminal = self.observation_manager.compute(update_history=False)['policy'].clone()
            s = mdp.state(self)
            for i in ids.tolist():
                self._sac_terminal[i] = {
                    'observation': terminal[i].detach().cpu().numpy().copy(),
                    'metrics': {k: bool(getattr(s, k)[i]) for k in mdp.METRICS},
                    'source': 'L5' if float(s.source[i, 1]) > -0.015 else 'L6',
                    'blocked_moves': int(s.blocked_moves[i]),
                    'failure': bool(self.termination_manager.get_term('failure')[i]),
                    'phase_success': bool(self.termination_manager.get_term('success')[i]),
                }
            self.recorder_manager.record_pre_reset(ids)
            self._reset_idx(ids)
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(ids)
        self.command_manager.compute(dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
