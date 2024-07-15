import time
import gymnasium
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple

from model import Policy
from rollout import Rollout


OptimizationResults = namedtuple(
    "OptimizationResults",
    [
        "learning_rate",
        "value_loss",
        "policy_loss",
        "entropy",
        "old_approx_kl",
        "approx_kl",
        "explained_variance",
    ],
)


class PPO:
    def __init__(
        self,
        policy: Policy,
        optimizer,
        env: gymnasium.Env,
        eval_env: gymnasium.Env,
        num_steps=128,
        gamma=0.99,
        learning_rate=3e-4,
        gae_lambda=0.95,
        batch_size=64,
        minibatch_size=16,
        update_epochs=4,
        clip_coef=0.2,
        norm_adv=True,
        clip_vloss=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.01,
        device="cpu",
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.env = env
        self.eval_env = eval_env
        self.num_steps = num_steps
        self.rollout_buffer = Rollout(self.num_steps, env, device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.update_epochs = update_epochs
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.clip_coef = clip_coef
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.device = device

    def _check_final_info(self, infos):
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info and "r" in info["episode"]:
                    self.rollout_buffer.add_episode_return(info["episode"]["r"])

    def collect_rollout(
        self,
        next_obs,
        next_done,
    ):
        self.rollout_buffer.reset()

        for _ in range(0, self.num_steps):
            next_obs, next_done = self._collect_step(next_obs, next_done)

        return next_obs, next_done

    def _collect_step(self, next_obs, next_done):
        obs = next_obs
        done = next_done

        with torch.no_grad():
            action, logprob, _, value = self.policy.get_action_and_value(next_obs)
            value = value.flatten()

        next_obs, reward, terminations, truncations, infos = self.env.step(
            action.cpu().numpy()
        )
        next_done = float(np.logical_or(terminations, truncations))
        reward = torch.tensor(reward).to(self.device).view(-1)
        if next_done:
            next_obs, _ = self.env.reset()
        next_obs = torch.Tensor(next_obs).to(self.device)

        self.rollout_buffer.add(
            obs,
            action,
            logprob,
            reward,
            done,
            value,
        )

        self._check_final_info(infos)
        return next_obs, next_done

    def _compute_advantages_returns(self, next_obs, next_done):
        with torch.no_grad():
            advantages = self._compute_advantages(next_obs, next_done)
            returns = advantages + self.rollout_buffer.values
        return advantages, returns

    def _compute_advantages(self, next_obs, next_done):
        next_value = self.policy.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(self.rollout_buffer.rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(self.rollout_buffer.step)):
            if t == self.rollout_buffer.step - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.rollout_buffer.dones[t + 1]
                nextvalues = self.rollout_buffer.values[t + 1]
            delta = (
                self.rollout_buffer.rewards[t]
                + self.gamma * nextvalues * nextnonterminal
                - self.rollout_buffer.values[t]
            )
            advantages[t] = lastgaelam = (
                delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            )

        return advantages

    def _anneal_learning_rate(self, iteration, num_iterations):
        frac = 1.0 - (iteration - 1.0) / num_iterations
        self.current_lr = frac * self.learning_rate
        self.optimizer.param_groups[0]["lr"] = self.current_lr

    def _flatten_data(self, advantages, returns):
        b_obs = self.rollout_buffer.obs.reshape(
            (-1,) + self.env.observation_space.shape
        )
        b_logprobs = self.rollout_buffer.logprobs.reshape(-1)
        b_actions = self.rollout_buffer.actions.reshape(
            (-1,) + self.env.action_space.shape
        )
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.rollout_buffer.values.reshape(-1)
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values

    def _optimize_epoch(
        self, b_inds, b_obs, b_actions, b_logprobs, b_returns, b_values, b_advantages
    ):
        np.random.shuffle(b_inds)
        for start in range(0, self.batch_size, self.minibatch_size):
            end = start + self.minibatch_size
            mb_inds = b_inds[start:end]

            mb_obs = b_obs[mb_inds]
            mb_actions = b_actions.long()[mb_inds]
            mb_logprobs = b_logprobs[mb_inds]
            mb_returns = b_returns[mb_inds]
            mb_values = b_values[mb_inds]
            mb_advantages = b_advantages[mb_inds]

            old_approx_kl, approx_kl, pg_loss, v_loss, entropy_loss = (
                self._calculate_policy_loss(
                    mb_obs,
                    mb_actions,
                    mb_logprobs,
                    mb_returns,
                    mb_values,
                    mb_advantages,
                )
            )
        return old_approx_kl, approx_kl, pg_loss, v_loss, entropy_loss

    def optimize(self, advantages, returns):
        b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = (
            self._flatten_data(advantages, returns)
        )
        b_inds = np.arange(self.batch_size)
        for epoch in range(self.update_epochs):
            old_approx_kl, approx_kl, pg_loss, v_loss, entropy_loss = (
                self._optimize_epoch(
                    b_inds,
                    b_obs,
                    b_actions,
                    b_logprobs,
                    b_returns,
                    b_values,
                    b_advantages,
                )
            )
            if self.target_kl is not None and approx_kl > self.target_kl:
                break
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return OptimizationResults(
            self.current_lr,
            v_loss.item(),
            pg_loss.item(),
            entropy_loss.item(),
            old_approx_kl.item(),
            approx_kl.item(),
            explained_var,
        )

    def _calculate_policy_loss(
        self,
        mb_obs,
        mb_actions,
        mb_logprobs,
        mb_returns,
        mb_values,
        mb_advantages,
    ):
        _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(
            mb_obs, mb_actions
        )
        logratio = newlogprob - mb_logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1.0) - logratio).mean()

        if self.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                mb_advantages.std() + 1e-8
            )

        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(
            ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef
        )
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        newvalue = newvalue.view(-1)
        if self.clip_vloss:
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_clipped = mb_values + torch.clamp(
                newvalue - mb_values,
                -self.clip_coef,
                self.clip_coef,
            )
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - self.ent_coef * entropy_loss + (v_loss / 2.0) * self.vf_coef

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return old_approx_kl, approx_kl, pg_loss, v_loss, entropy_loss

    def learn(self, num_iterations, seed, anneal_lr=True):
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.env.reset(seed=seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = 0

        results = []
        checkpoint = num_iterations // 10
        for global_step in range(0, num_iterations, self.num_steps):
            if anneal_lr:
                self._anneal_learning_rate(global_step, num_iterations)
            next_obs, next_done = self.collect_rollout(
                next_obs,
                next_done,
            )
            advantages, returns = self._compute_advantages_returns(
                next_obs,
                next_done,
            )
            result = self.optimize(advantages, returns)
            results.append(result)
            if global_step >= checkpoint:
                checkpoint += num_iterations // 10
                mean_epi_return = np.nanmean(self.evaluate())
                print(f"step {global_step}/{num_iterations} mean: {mean_epi_return}")
                if mean_epi_return > self.eval_env.spec.reward_threshold:
                    print("Threshold reached!")
                    break
        end_time = time.time()
        print(f"Learning took {end_time - start_time} seconds")
        return results

    def evaluate(self, num_episodes=10):
        rewards = []
        for _ in range(num_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            obs = torch.Tensor(obs).to(self.device)
            episode_reward = 0
            while not done:
                with torch.no_grad():
                    action = self.policy.get_best_action(obs)
                obs, reward, done, terminated, _ = self.eval_env.step(
                    action.cpu().numpy()
                )
                done = done or terminated
                obs = torch.Tensor(obs).to(self.device)
                episode_reward += reward
            rewards.append(episode_reward)
        return rewards
