"""Dual-PSM peg transfer from L5/L6 to R2."""

import gymnasium as gym

from .agents.rsl_rl_cfg import PegTransferPPORunnerCfg
from .env_cfg import PegTransferEnvCfg, PegTransferEnvCfg_PLAY


gym.register(
    id="Isaac-Peg-Transfer-Dual-PSM-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": PegTransferEnvCfg,
        "rsl_rl_cfg_entry_point": PegTransferPPORunnerCfg,
    },
    disable_env_checker=True,
)

# SAC uses a distinct task, observation schema, continuous jaws and guarded geometry.
# Import its configuration lazily: existing PPO entry points retain their settings.
gym.register(
    id="Isaac-Peg-Transfer-Dual-PSM-SAC-v0",
    entry_point=__name__ + ".sac_env:PegTransferSACEnv",
    kwargs={"env_cfg_entry_point": __name__ + ".sac_env_cfg:PegTransferSACEnvCfg"},
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Peg-Transfer-Dual-PSM-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": PegTransferEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": PegTransferPPORunnerCfg,
    },
    disable_env_checker=True,
)
