import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass
from ..peg_transfer.sac_env_cfg import PegTransferSACEnvCfg, Events as PhysicalEvents
from . import mdp

@configclass
class Events(PhysicalEvents):
    reset_sequence = EventTermCfg(func=mdp.reset, mode='reset')

@configclass
class Observations:
    @configclass
    class Policy(ObservationGroupCfg):
        state = ObservationTermCfg(func=mdp.observation)
        enable_corruption = False
        concatenate_terms = True
    policy: Policy = Policy()

@configclass
class Rewards:
    sequence = RewardTermCfg(func=mdp.reward, weight=1.)

@configclass
class Terminations:
    time_out = TerminationTermCfg(func=base_mdp.time_out, time_out=True)
    failure = TerminationTermCfg(func=mdp.failure)
    success = TerminationTermCfg(func=mdp.success)

@configclass
class SequenceEnvCfg(PegTransferSACEnvCfg):
    events: Events = Events()
    observations: Observations = Observations()
    rewards: Rewards = Rewards()
    terminations: Terminations = Terminations()
    skill: str = 'full'
    guidance: float = 1.
    residual_translation: float = .0001
    sequence_gamma: float = .999
    stage_budget: int = 600
    def __post_init__(self):
        super().__post_init__()
        self.phase = 'full'  # Never terminate on a legacy lift before the selected prefix.
        self.episode_length_s = 60.
