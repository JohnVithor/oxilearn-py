import torch
from small_rng_rs import SmallRng
import numpy as np
from torch import tensor


class ExperienceStats:
    def __init__(self, shape, device):
        self.means = torch.zeros(shape, dtype=torch.float64, device=device)
        self.msqs = torch.ones(shape, dtype=torch.float64, device=device)
        self.count = torch.zeros(1, dtype=torch.int64, device=device)

    def push(self, value):
        self.count += 1
        delta = value - self.means
        self.means += delta / self.count
        delta2 = value - self.means
        self.msqs += delta * delta2

    def mean(self):
        return self.means

    def var(self):
        return self.msqs / (self.count.item() - 1)


class RandomExperienceBuffer:
    def __init__(self, capacity, obs_size, minsize, seed, normalize_obs, device):
        self.obs_size = obs_size
        self.curr_states = torch.empty(
            (capacity, obs_size), dtype=torch.float64, device=device
        )
        self.curr_actions = torch.empty(capacity, dtype=torch.int64, device=device)
        self.rewards = torch.empty(capacity, dtype=torch.float64, device=device)
        self.next_states = torch.empty(
            (capacity, obs_size), dtype=torch.float64, device=device
        )
        self.dones = torch.empty(capacity, dtype=torch.int8, device=device)
        self.capacity = capacity
        self.next_idx = 0
        self.size = 0
        self.minsize = minsize
        self.rng = SmallRng(seed)
        self.device = device
        self.stats = ExperienceStats(obs_size, device)
        self.normalize_obs = normalize_obs

    def ready(self):
        return self.size >= self.minsize

    def add(self, curr_state, curr_action, reward, done, next_state):
        curr_state = curr_state.to(self.device)
        curr_action = tensor(curr_action, dtype=torch.int64, device=self.device)
        reward = tensor(reward, dtype=torch.float64, device=self.device)
        done = tensor(done, dtype=torch.int8, device=self.device)
        next_state = next_state.to(self.device)

        self.curr_states[self.next_idx] = curr_state
        self.next_states[self.next_idx] = next_state

        self.curr_actions[self.next_idx] = curr_action
        self.rewards[self.next_idx] = reward
        self.dones[self.next_idx] = done

        self.next_idx = (self.next_idx + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

        if self.normalize_obs:
            self.stats.push(curr_state)

    def normalize(self, values):
        if self.normalize_obs:
            return (values - self.stats.mean()) / torch.sqrt(self.stats.var())
        else:
            return values

    def sample_batch(self, size):
        indices = self.rng.integers(0, self.size, size)
        indices = tensor(indices, dtype=torch.int64, device=self.device)
        return (
            self.normalize(self.curr_states[indices]).float(),
            self.curr_actions[indices].reshape(-1, 1),
            self.rewards[indices].reshape(-1, 1),
            self.dones[indices].reshape(-1, 1),
            self.normalize(self.next_states[indices]).float(),
        )
