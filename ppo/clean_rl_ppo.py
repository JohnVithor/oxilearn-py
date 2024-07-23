# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical

from rollout import Rollout


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 10000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


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


def evaluate(agent, eval_env, device, num_episodes=10):
    rewards = []
    for _ in range(num_episodes):
        obs, _ = eval_env.reset()
        done = False
        obs = torch.Tensor(obs).to(device)
        episode_reward = 0
        while not done:
            with torch.no_grad():
                action = agent.get_best_action(obs)
            obs, reward, done, terminated, _ = eval_env.step(action.cpu().numpy())
            done = done or terminated
            obs = torch.Tensor(obs).to(device)
            episode_reward += reward
        rewards.append(episode_reward)
    return rewards


def collect_rollout(
    args, device, envs, agent, rollout_buffer, global_step, next_obs, next_done
):
    rollout_buffer.reset()

    for step in range(0, args.num_steps):
        global_step += args.num_envs
        obs_step = next_obs
        dones_step = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values_step = value.flatten()
        actions_step = action
        logprobs_step = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, terminations, truncations, infos = envs.step(
            action.cpu().numpy()
        )
        next_done = np.logical_or(terminations, truncations)
        rewards_step = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
            next_done
        ).to(device)

        rollout_buffer.add(
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
                    rollout_buffer.add_episode_return(info["episode"]["r"])
                    # print(
                    #     f"global_step={global_step}, episode_return={info['episode']['r']}"
                    # )
    return next_obs, next_done


def compute_advantages(agent, rollout_buffer: Rollout, args: Args, next_obs, next_done):
    next_value = agent.get_value(next_obs).reshape(1, -1)
    advantages = torch.zeros_like(rollout_buffer.rewards).to(device)
    lastgaelam = 0
    for t in reversed(range(rollout_buffer.step)):
        if t == rollout_buffer.step - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - rollout_buffer.dones[t + 1]
            nextvalues = rollout_buffer.values[t + 1]
        delta = (
            rollout_buffer.rewards[t]
            + args.gamma * nextvalues * nextnonterminal
            - rollout_buffer.values[t]
        )
        advantages[t] = lastgaelam = (
            delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        )

    return advantages


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(args.env_id))
            for i in range(args.num_envs)
        ],
    )
    eval_env = gym.make(args.env_id)
    eval_env.reset(seed=args.seed)
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = Policy(
        np.array(envs.single_observation_space.shape).prod(), envs.single_action_space.n
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    rollout_buffer = Rollout(args.num_steps, args.num_envs, envs, device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        next_obs, next_done = collect_rollout(
            args, device, envs, agent, rollout_buffer, global_step, next_obs, next_done
        )

        with torch.no_grad():
            advantages = compute_advantages(
                agent, rollout_buffer, args, next_obs, next_done
            )
            returns = advantages + rollout_buffer.values

        # flatten the batch
        b_obs = rollout_buffer.obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = rollout_buffer.logprobs.reshape(-1)
        b_actions = rollout_buffer.actions.reshape(
            (-1,) + envs.single_action_space.shape
        )
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = rollout_buffer.values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        eval_rewards = evaluate(agent, eval_env, device)
        mean_epi_return = np.nanmean(eval_rewards)
        print(f"step {global_step}/{iteration} mean: {mean_epi_return}")

    envs.close()
