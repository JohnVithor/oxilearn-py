import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
from small_rng_rs import SmallRng
from dqn import DoubleDeepAgent, generate_policy
from epsilon_greedy import EpsilonGreedy, EpsilonUpdateStrategy
from experience_buffer import RandomExperienceBuffer
from trainer import Trainer, TrainResults
from cart_pole import CartPoleEnv


class DQN:
    def __init__(
        self,
        net_arch,
        learning_rate,
        last_activation="none",
        memory_size=5000,
        min_memory_size=1000,
        discount_factor=0.99,
        initial_epsilon=1.0,
        final_epsilon=0.05,
        exploration_fraction=0.05,
        max_grad_norm=10.0,
        seed=0,
        normalize_obs=False,
        optimizer="Adam",
        loss_fn="MSE",
        device="cpu",
    ):
        self.net_arch = net_arch
        self.learning_rate = learning_rate
        self.last_activation = self.get_activation(last_activation)
        self.memory_size = memory_size
        self.min_memory_size = min_memory_size
        self.discount_factor = discount_factor
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.exploration_fraction = exploration_fraction
        self.max_grad_norm = max_grad_norm
        self.normalize_obs = normalize_obs
        self.agent = None
        self.rng = SmallRng(seed)
        torch.manual_seed(self.rng.next_u64())
        self.optimizer_name = optimizer
        self.loss_fn_name = loss_fn
        self.optimizer = self.get_optimizer(optimizer)
        self.loss_fn = self.get_loss_fn(loss_fn)
        self.device = device

    @staticmethod
    def get_activation(id):
        return {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "softmax": nn.Softmax(dim=0),
            "tanh": nn.Tanh(),
        }.get(id, nn.Identity())

    @staticmethod
    def get_optimizer(optimizer):
        return {
            "Adam": optim.Adam,
            "Sgd": optim.SGD,
            "RmsProp": optim.RMSprop,
            "AdamW": optim.AdamW,
        }.get(optimizer, None)

    @staticmethod
    def get_loss_fn(loss_fn):
        return {
            "MAE": nn.L1Loss(),
            "MSE": nn.MSELoss(),
            "RMSE": lambda x, y: torch.sqrt(nn.MSELoss()(x, y)),
            "Huber": nn.HuberLoss(),
            "smooth_l1": nn.SmoothL1Loss(),
        }.get(loss_fn, None)

    def reset(self):
        self.agent = None

    def save(self, path):
        if self.agent:
            self.agent.save_net(path)
        else:
            raise ValueError("Agent not initialized!")

    def load(self, path):
        if self.agent:
            self.agent.load_net(path)
        else:
            raise ValueError("Agent not initialized!")

    def prepare(self, environment):
        self.create_agent(environment)

    def create_agent(self, environment):
        input_size = environment.observation_space.shape[0]
        output_size = environment.action_space.n

        mem_replay = RandomExperienceBuffer(
            self.memory_size,
            input_size,
            self.min_memory_size,
            self.rng.next_u64(),
            self.normalize_obs,
            self.device,
        )
        update_strategy = EpsilonUpdateStrategy.EpsilonLinearTrainingDecreasing(
            start=self.initial_epsilon,
            end=self.final_epsilon,
            end_fraction=self.exploration_fraction,
        )
        action_selector = EpsilonGreedy(
            self.initial_epsilon, self.rng.next_u64(), update_strategy
        )
        arch = [(size, self.get_activation(f)) for size, f in self.net_arch]

        policy = generate_policy(
            arch,
            nn.Identity(),
            input_size,
            output_size,
        )

        self.agent = DoubleDeepAgent(
            action_selector,
            mem_replay,
            policy,
            self.optimizer,
            self.loss_fn,
            self.learning_rate,
            self.discount_factor,
            self.max_grad_norm,
            self.device,
        )

    def train(
        self,
        env,
        eval_env,
        solve_with,
        steps=50_000,
        gradient_steps=175,
        train_freq=200,
        update_freq=10,
        batch_size=128,
        eval_freq=1000,
        eval_for=10,
        verbose=0,
    ):
        if self.agent is None:
            self.create_agent(env)

        trainer = Trainer(env, eval_env)
        trainer.early_stop = lambda reward: reward >= solve_with

        return trainer.train_by_steps(
            self.agent,
            steps,
            gradient_steps,
            train_freq,
            batch_size,
            update_freq,
            eval_freq,
            eval_for,
            verbose,
        )

    def evaluate(self, env, n_eval_episodes):
        if self.agent is None:
            self.create_agent(env)

        trainer = Trainer(env, env)
        evaluation_results = trainer.evaluate(self.agent, n_eval_episodes)
        rewards = evaluation_results[0]
        reward_avg = sum(rewards) / len(rewards)
        variance = sum((reward_avg - value) ** 2 for value in rewards) / len(rewards)
        std = variance**0.5
        return variance, std
