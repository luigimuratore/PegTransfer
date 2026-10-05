import torch
import gymnasium as gym
from ..peg_transfer.sac_env import PegTransferSACEnv
from . import mdp, assistance

class SequenceEnv(PegTransferSACEnv):
    def __init__(self, *args, **kwargs):
        self._sequence_pending = {}
        super().__init__(*args, **kwargs)
        self._sequence_action = torch.zeros(self.num_envs, 4, device=self.device)
        self.single_action_space = gym.spaces.Box(-1., 1., (4,), dtype='float32')
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def step(self, action):
        self._sequence_pending = {}
        self._sac_terminal = {}
        self._sequence_action = action.to(self.device).clamp(-1, 1)
        self.action_manager.process_action(mdp.motor_action(self, self._sequence_action))
        self.recorder_manager.record_pre_step()
        rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        assistance.begin_step(self)
        # Same locally verified Isaac Lab step order as the legacy environment,
        # with sequence-only pose assistance. Legacy SAC sources stay untouched.
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            assistance.assist(self)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and rendering:
                self.sim.render()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        mdp.physical.update(self)
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        if self.recorder_manager.active_terms:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()
        ids = self.reset_buf.nonzero().flatten()
        if len(ids):
            terminal = self.observation_manager.compute(update_history=False)['policy'].clone()
            s = mdp.physical.state(self)
            for i in ids.tolist():
                summary = self._sequence_pending[i]
                self._sac_terminal[i] = dict(observation=terminal[i].detach().cpu().numpy().copy(),
                    metrics={k: bool(getattr(s, k)[i]) for k in mdp.physical.METRICS},
                    sequence=summary, source=mdp.g.source_label(s.source[i:i+1, :2])[0],
                    blocked_moves=int(s.blocked_moves[i]),
                    failure=bool(self.termination_manager.get_term('failure')[i]),
                    phase_success=bool(self.termination_manager.get_term('success')[i]))
                self._sac_terminal[i]['metrics']['full_success'] &= summary['stage'] >= 20
            self.recorder_manager.record_pre_reset(ids)
            self._reset_idx(ids)
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()
            self.recorder_manager.record_post_reset(ids)
        self.command_manager.compute(dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
