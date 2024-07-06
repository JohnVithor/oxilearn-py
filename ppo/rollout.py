import torch


class Rollout:

    def __init__(self, num_steps: int, num_envs: int, envs, device: str) -> None:
        self.obs = torch.zeros(
            (num_steps, num_envs) + envs.single_observation_space.shape
        ).to(device)
        self.actions = torch.zeros(
            (num_steps, num_envs) + envs.single_action_space.shape
        ).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs)).to(device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)

        self.step = 0

    def add(self, obs, action, logprob, reward, done, value) -> None:
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value

        self.step += 1

    def reset(self) -> None:
        self.step = 0

    def get_done_rewards(self):
        dones = self.dones.sum(dim=0)
        return self.rewards.sum(dim=0) * dones / dones.sum()
