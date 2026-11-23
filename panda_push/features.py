"""Goal-relative features are derived after HER substitutes desired_goal."""

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PushFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, observation_space["observation"].shape[0] + 9)
        self.register_buffer("origin", torch.tensor([0.5, 0, 0.24]))

    def forward(self, observations):
        cube = observations["achieved_goal"]
        goal = observations["desired_goal"]
        return torch.cat((observations["observation"], 5 * (cube - self.origin),
                          5 * (goal - self.origin), 5 * (goal - cube)), dim=-1)

