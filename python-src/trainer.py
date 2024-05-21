import torch
import numpy as np
from typing import Tuple, List

TrainResults = Tuple[List[float], List[int], List[float], List[float], List[float]]


class Trainer:
    def __init__(self, env, eval_env):
        self.env = env
        self.eval_env = eval_env
        self.early_stop = None  # Optional[Callable[[float], bool]]

    def train_by_steps(
        self,
        agent,
        n_steps: int,
        gradient_steps: int,
        train_freq: int,
        batch_size: int,
        update_freq: int,
        eval_freq: int,
        eval_for: int,
        verbose: int,
    ) -> TrainResults:
        data, _ = self.env.reset()
        curr_obs = torch.tensor(data, dtype=torch.float32)
        training_reward = []
        training_length = []
        training_error = []
        evaluation_reward = []
        evaluation_length = []

        n_episodes = 1
        action_counter = 0
        epi_reward = 0.0
        agent.reset()

        for step in range(1, n_steps + 1):
            action_counter += 1
            curr_action = agent.get_action(curr_obs)
            next_obs, reward, done, truncated, _ = self.env.step(curr_action)
            next_obs = torch.tensor(next_obs, dtype=torch.float32)

            epi_reward += reward
            agent.add_transition(curr_obs, curr_action, reward, done, next_obs)

            curr_obs = next_obs

            if step % train_freq == 0:
                td_error = agent.update(gradient_steps, batch_size)
                if td_error is not None:
                    training_error.append(td_error)

            if done or truncated:
                training_reward.append(epi_reward)
                training_length.append(action_counter)
                if n_episodes % update_freq == 0:
                    if not agent.update_networks():
                        print("copy error")
                data, _ = self.env.reset()
                curr_obs = torch.tensor(data, dtype=torch.float32)
                agent.action_selection_update(step / n_steps, epi_reward)
                n_episodes += 1
                epi_reward = 0.0
                action_counter = 0

            if step % eval_freq == 0:
                rewards, eval_lengths = self.evaluate(agent, eval_for)
                reward_avg = np.mean(rewards)
                eval_lengths_avg = np.mean(eval_lengths)
                if verbose > 0:
                    print(
                        f"steps number: {step} - eval reward: {reward_avg} - epsilon: {agent.get_epsilon()}"
                    )
                evaluation_reward.append(reward_avg)
                evaluation_length.append(eval_lengths_avg)
                if self.early_stop and self.early_stop(reward_avg):
                    training_reward.append(epi_reward)
                    training_length.append(action_counter)
                    break

        return (
            training_reward,
            training_length,
            training_error,
            evaluation_reward,
            evaluation_length,
        )

    def evaluate(self, agent, n_episodes: int) -> Tuple[List[float], List[int]]:
        reward_history = []
        episode_length = []
        for _ in range(n_episodes):
            epi_reward = 0.0
            data, _ = self.eval_env.reset()
            obs_repr = torch.tensor(data, dtype=torch.float32)
            curr_action = agent.get_best_action(obs_repr)
            action_counter = 0
            while True:
                obs, reward, done, truncated, _ = self.eval_env.step(curr_action)
                next_obs_repr = torch.tensor(obs, dtype=torch.float32)
                next_action_repr = agent.get_best_action(next_obs_repr)
                curr_action = next_action_repr
                epi_reward += reward
                if done or truncated:
                    reward_history.append(epi_reward)
                    episode_length.append(action_counter)
                    break
                action_counter += 1
        return reward_history, episode_length
