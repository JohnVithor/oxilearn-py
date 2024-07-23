import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class Policy(nn.Module):
    def __init__(self, obs_shape, num_actions):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    def get_best_action(self, x):
        logits = self.actor(x)
        return torch.argmax(logits, dim=-1)
