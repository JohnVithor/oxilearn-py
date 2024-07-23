import gymnasium
import torch


class Rollout:

    def __init__(self, num_steps: int, env: gymnasium.Env, device: str) -> None:
        self.obs = torch.zeros((num_steps,) + env.observation_space.shape).to(device)
        self.actions = torch.zeros((num_steps,) + env.action_space.shape).to(device)
        self.logprobs = torch.zeros((num_steps)).to(device)
        self.rewards = torch.zeros((num_steps)).to(device)
        self.dones = torch.zeros((num_steps)).to(device)
        self.values = torch.zeros((num_steps)).to(device)

        self.episode_returns = []

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
        self.episode_returns = []

    def add_episode_return(self, episode_return: float) -> None:
        self.episode_returns.append(episode_return)
