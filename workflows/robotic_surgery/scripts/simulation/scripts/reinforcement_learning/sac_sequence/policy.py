"""Actor depends on measured goal error, jaws and stage, not a learned clock.

The critic retains the full 112-dimensional Markov state. The actor sees 29
features: body-frame target error, jaw measurements, stage, active arm, guidance.
No nominal/expert action is provided as an input at inference.
"""
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import SACPolicy

class GoalExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        if observation_space.shape != (112,):
            raise ValueError('GoalExtractor requires sequence-v1 observation schema (112,)')
        super().__init__(observation_space,29)
        self.register_buffer('indices',torch.tensor([0,1,2,60,61,*range(84,108)],dtype=torch.long))
    def forward(self, observations):
        return observations.index_select(1,self.indices)

class GoalPolicy(SACPolicy):
    def make_actor(self, features_extractor=None):
        return super().make_actor(GoalExtractor(self.observation_space))
