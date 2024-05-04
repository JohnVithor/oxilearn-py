import sys
import torch
import numpy as np
import random
import os

import gymnasium as gym
from oxilearn import DQNAgent
import optuna
from optuna import Trial


def create_objective(env_name, seed, verbose):

    def objective(trial: Trial) -> float:

        eval_size = 10
        eval_freq = 1_000

        env = gym.make("CartPole-v1")
        eval_env = gym.make("CartPole-v1")

        learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.1)
        batch_size = trial.suggest_int("batch_size", 16, 256)
        buffer_size = trial.suggest_int("buffer_size", 1_000, 100_000)
        min_buffer_size = trial.suggest_int("min_buffer_size", 1_000, 10_000)
        gamma = trial.suggest_float("gamma", 0.9, 0.99)
        target_update_interval = trial.suggest_int("target_update_interval", 1, 256)
        train_freq = trial.suggest_int("train_freq", 1, 256)
        gradient_steps = trial.suggest_int("gradient_steps", 1, 256)
        exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.25)
        exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.0, 0.1)
        optimizer = trial.suggest_categorical(
            "optimizer", ["Adam", "AdamW", "RmsProp", "Sgd"]
        )
        loss_fn = trial.suggest_categorical(
            "loss_fn", ["MAE", "MSE", "RMSE", "Huber", "smooth_l1"]
        )
        max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 10.0)

        agent = DQNAgent(
            net_arch=[(256, "relu"), (256, "relu")],
            learning_rate=learning_rate,
            last_activation="none",
            memory_size=buffer_size,
            min_memory_size=min_buffer_size,
            discount_factor=gamma,
            initial_epsilon=1.0,
            final_epsilon=exploration_final_eps,
            exploration_fraction=exploration_fraction,
            max_grad_norm=max_grad_norm,
            seed=seed,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )
        agent.prepare(env)

        env.reset(seed=seed + 1)
        eval_env.reset(seed=seed + 2)

        results = agent.train(
            env,
            eval_env,
            env.spec.reward_threshold,
            steps=1_000_000,
            gradient_steps=gradient_steps,
            train_freq=train_freq,
            update_freq=target_update_interval,
            batch_size=batch_size,
            eval_at=eval_freq,
            eval_for=eval_size,
            verbose=verbose,
        )
        training_steps = sum(results[1])
        reward, std = agent.evaluate(eval_env, eval_size)

        return training_steps, reward

    return objective


if __name__ == "__main__":
    env_name = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 1 else 0
    verbose = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    os.environ["PYTHONASHSEED"] = f"{seed}"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    study_name = f"rust_{env_name}"
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        directions=["minimize", "maximize"],
        load_if_exists=True,
    )
    study.optimize(
        create_objective(env_name, seed, verbose),
        n_trials=100,
        n_jobs=2,
        show_progress_bar=True,
    )
    print(study.best_params)
