"""Versioned, goal-conditioned SAC sequence; legacy tasks remain unchanged."""
import gymnasium as gym

gym.register(id='Isaac-Peg-Transfer-Dual-PSM-Sequence-v1',
             entry_point=__name__ + '.env:SequenceEnv',
             kwargs={'env_cfg_entry_point': __name__ + '.env_cfg:SequenceEnvCfg'},
             disable_env_checker=True)
