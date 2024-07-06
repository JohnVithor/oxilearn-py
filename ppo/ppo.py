import time
import numpy as np
import torch
import torch.nn as nn

from model import Policy
from rollout import Rollout


class PPO:
    def __init__(
        self,
        policy: Policy,
        optimizer,
        envs,
        num_steps=128,
        num_envs=1,
        gamma=0.99,
        gae_lambda=0.95,
        batch_size=32,
        minibatch_size=4,
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
        self.envs = envs
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.rollout_buffer = Rollout(num_steps, num_envs, envs, device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.update_epochs = update_epochs
        self.learning_rate = self.optimizer.param_groups[0]["lr"]
        self.clip_coef = clip_coef
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.device = device

    def collect_rollout(
        self,
        global_step,
        next_obs,
        next_done,
    ):
        self.rollout_buffer.reset()

        for step in range(0, self.num_steps):
            global_step += self.num_envs
            obs_step = next_obs
            dones_step = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = self.policy.get_action_and_value(next_obs)
                values_step = value.flatten()
            actions_step = action
            logprobs_step = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = self.envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards_step = torch.tensor(reward).to(self.device).view(-1)
            next_obs = torch.Tensor(next_obs).to(self.device)
            next_done = torch.Tensor(next_done).to(self.device)

            self.rollout_buffer.add(
                obs_step,
                actions_step,
                logprobs_step,
                rewards_step,
                dones_step,
                values_step,
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
            #             print("charts/episodic_return", info["episode"]["r"], global_step)
            #             print("charts/episodic_length", info["episode"]["l"], global_step)

        return next_obs, next_done, global_step

    def compute_advantages_returns(self, next_obs, next_done):
        with torch.no_grad():
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
            returns = advantages + self.rollout_buffer.values
        return advantages, returns

    def anneal_learning_rate(self, iteration, num_iterations):
        frac = 1.0 - (iteration - 1.0) / num_iterations
        self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

    def _flatten_data(self, advantages, returns):
        b_obs = self.rollout_buffer.obs.reshape(
            (-1,) + self.envs.single_observation_space.shape
        )
        b_logprobs = self.rollout_buffer.logprobs.reshape(-1)
        b_actions = self.rollout_buffer.actions.reshape(
            (-1,) + self.envs.single_action_space.shape
        )
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.rollout_buffer.values.reshape(-1)
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values

    def optimize(self, advantages, returns):
        b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = (
            self._flatten_data(advantages, returns)
        )
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - self.ent_coef * entropy_loss
                    + (v_loss / 2.0) * self.vf_coef
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return self.optimizer.param_groups[0]["lr"]

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # print("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # print("losses/value_loss", v_loss.item(), global_step)
        # print("losses/policy_loss", pg_loss.item(), global_step)
        # print("losses/entropy", entropy_loss.item(), global_step)
        # print("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # print("losses/approx_kl", approx_kl.item(), global_step)
        # print("losses/clipfrac", np.mean(clipfracs), global_step)
        # print("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        # print("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    def learn(self, num_iterations, seed, anneal_lr=True):
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset(seed=seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)

        while global_step < num_iterations:
            # print(f"Iteration {iteration}/{num_iterations}")
            if anneal_lr:
                self.anneal_learning_rate(global_step, num_iterations)
            next_obs, next_done, global_step = self.collect_rollout(
                global_step,
                next_obs,
                next_done,
            )
            advantages, returns = self.compute_advantages_returns(
                next_obs,
                next_done,
            )
            self.optimize(advantages, returns)
            # if iteration % 10 == 0:
            #     mean_rewards = self.rollout_buffer.get_done_rewards()
            #     print(
            #         f"Iteration {iteration}/{num_iterations} mean rewards: {mean_rewards}"
            #     )
        end_time = time.time()
        print(f"Learning took {end_time - start_time} seconds")
