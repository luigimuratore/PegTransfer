"""Separate SAC configuration: legacy PPO observations/actions remain untouched."""

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.managers import RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass

from . import sac_geometry as g, sac_mdp
from .env_cfg import PegTransferEnvCfg
from .sac_actions import ContinuousJawActionCfg
from .sac_assets import spawn_hollow_peg


@configclass
class Events:
    reset_all = EventTermCfg(func=base_mdp.reset_scene_to_default, mode='reset')
    reset_transfer = EventTermCfg(func=sac_mdp.reset, mode='reset')


@configclass
class Observations:
    @configclass
    class Policy(ObservationGroupCfg):
        state = ObservationTermCfg(func=sac_mdp.observation)
        enable_corruption = False
        concatenate_terms = True
    policy: Policy = Policy()


@configclass
class Rewards:
    transfer = RewardTermCfg(func=sac_mdp.reward, weight=1.0)


@configclass
class Terminations:
    time_out = TerminationTermCfg(func=base_mdp.time_out, time_out=True)
    failure = TerminationTermCfg(func=sac_mdp.failure)
    success = TerminationTermCfg(func=sac_mdp.success)


@configclass
class PegTransferSACEnvCfg(PegTransferEnvCfg):
    events: Events = Events()
    observations: Observations = Observations()
    rewards: Rewards = Rewards()
    terminations: Terminations = Terminations()
    phase: str = 'full'
    source_post: str = 'L5'

    def __post_init__(self):
        super().__post_init__()
        self.sim.physx.enable_ccd = True
        self.scene.object.spawn.func = spawn_hollow_peg
        self.scene.object.spawn.rigid_props.solver_position_iteration_count = 16
        self.scene.object.spawn.rigid_props.solver_velocity_iteration_count = 4
        self.scene.object.init_state.pos = (-g.HOLE_CENTER[0], 0.0, g.RESET_Z)
        for a in (1, 2):
            setattr(self.actions, f'finger_{a}_joint_pos', ContinuousJawActionCfg(asset_name=f'robot_{a}'))
            tool = getattr(self.scene, f'robot_{a}').actuators['psm_tool']
            tool.velocity_limit = None
            tool.velocity_limit_sim = 0.8
        self.episode_length_s = 40.0


def configure_phase(cfg, phase):
    cfg.phase = phase
    cfg.episode_length_s = {'lift': 16.0, 'handover': 30.0, 'full': 40.0}[phase]
