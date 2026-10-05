"""One continuous jaw command per PSM, suitable for a squashed Gaussian SAC actor."""

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class ContinuousJawAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.ids, names = self._asset.find_joints("psm_tool_gripper.*_joint")
        if len(names) != 2:
            raise ValueError(f"Expected two jaw joints, found {names}")
        self.signs = torch.tensor([-1.0 if 'gripper1' in n else 1.0 for n in names], device=env.device)
        self._raw = torch.zeros(env.num_envs, 1, device=env.device)
        self._targets = torch.zeros(env.num_envs, 2, device=env.device)

    @property
    def action_dim(self):
        return 1

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._targets

    def process_actions(self, actions):
        self._raw[:] = actions.clamp(-1, 1)
        # -1 = closed (0.07 rad); +1 = open (0.5 rad); 0 = half open.
        angle = 0.07 + 0.43 * (self._raw + 1) / 2
        self._targets[:] = angle * self.signs

    def apply_actions(self):
        self._asset.set_joint_position_target(self._targets, joint_ids=self.ids)

    def reset(self, env_ids=None):
        self._raw[env_ids] = 0


@configclass
class ContinuousJawActionCfg(ActionTermCfg):
    class_type: type = ContinuousJawAction
