import gymnasium as gym
import sys
import torch
from torch import nn
from dqn import DoubleDeepAgent, generate_policy
from epsilon_greedy import EpsilonGreedy, EpsilonUpdateStrategy
from experience_buffer import RandomExperienceBuffer
from trainer import Trainer, TrainResults
from cart_pole import CartPoleEnv
import os
import optuna
from optuna import Trial
import numpy as np
import random

least_steps_number = 100_000


def create_objective(seed, verbose):

    def objective(trial: Trial) -> float:
        device = "cpu"
        eval_size = 10
        eval_freq = 1_000

        train_env = CartPoleEnv()
        train_env.reset(seed=seed)

        eval_env = CartPoleEnv()
        eval_env.reset(seed=seed + 1)

        learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.1)
        batch_size = trial.suggest_int("batch_size", 16, 256)
        buffer_size = trial.suggest_int("buffer_size", 1_000, 100_000)
        min_buffer_size = trial.suggest_int("min_buffer_size", 1_000, 10_000)
        target_update_interval = trial.suggest_int("target_update_interval", 1, 256)
        train_freq = trial.suggest_int("train_freq", 1, 256)
        gradient_steps = trial.suggest_int("gradient_steps", 1, 256)
        exploration_initial_eps = trial.suggest_float(
            "exploration_initial_eps", 0.5, 1.0
        )
        exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.25)
        exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.0, 0.1)
        # optimizer = trial.suggest_categorical(
        #     "optimizer",
        #     [torch.optim.Adam, torch.optim.AdamW, torch.optim.RMSprop, torch.optim.SGD],
        # )
        # loss_fn = trial.suggest_categorical(
        #     "loss_fn", [nn.MSELoss(), nn.HuberLoss(), nn.SmoothL1Loss()]
        # )

        update_strategy = EpsilonUpdateStrategy.EpsilonLinearTrainingDecreasing(
            start=exploration_initial_eps,
            end=exploration_final_eps,
            end_fraction=exploration_fraction,
        )
        action_selector = EpsilonGreedy(
            exploration_initial_eps, seed + 2, update_strategy
        )

        mem_replay = RandomExperienceBuffer(
            buffer_size, 4, min_buffer_size, seed + 3, True, device
        )

        policy = generate_policy(
            [(256, nn.ReLU()), (256, nn.ReLU())],
            nn.Identity(),
            4,
            2,
        )

        model = DoubleDeepAgent(
            action_selector,
            mem_replay,
            policy,
            torch.optim.Adam,
            nn.MSELoss(),
            learning_rate,
            0.99,
            1.0,
            device,
        )

        trainer = Trainer(train_env, eval_env)
        trainer.early_stop = lambda reward: reward >= 475.0

        training_steps = 0
        for i, steps in enumerate(range(10_000, 50_000, 5_000)):
            training_results = trainer.train_by_steps(
                model,
                steps,
                gradient_steps,
                train_freq,
                batch_size,
                target_update_interval,
                1000,
                10,
                verbose,
            )
            training_steps = sum(training_results[1])
            if training_steps >= least_steps_number:
                evaluation_results = trainer.evaluate(model, 10)
                rewards = evaluation_results[0]
                reward_avg = sum(rewards) / len(rewards)
                return training_steps, reward_avg
        rewards = evaluation_results[0]
        reward_avg = sum(rewards) / len(rewards)
        return training_steps, reward_avg

    return objective


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    verbose = int(sys.argv[2]) if len(sys.argv) > 3 else 0

    os.environ["PYTHONASHSEED"] = f"{seed}"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    study_name = f"python_cart_pole"
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        directions=["minimize", "maximize"],
        load_if_exists=True,
    )
    study.optimize(
        create_objective(seed, verbose),
        n_trials=100,
        n_jobs=-1,
        show_progress_bar=True,
    )
    print(study.best_params)
