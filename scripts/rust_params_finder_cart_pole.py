import sys
import torch
import numpy as np
import random
import os

import gymnasium as gym
from oxilearn import DQN
import optuna
from optuna import Trial

least_steps_number = 100_000


def create_objective(env_name, seed, verbose):

    def objective(trial: Trial) -> float:

        eval_size = 10
        eval_freq = 1_000

        env = gym.make("CartPole-v1")
        eval_env = gym.make("CartPole-v1")

        learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.1)
        batch_size = trial.suggest_int("batch_size", 16, 256)
        buffer_size = trial.suggest_int("buffer_size", 1_000, 100_000)
        min_buffer_size = trial.suggest_int("min_buffer_size", 256, 10_000)
        target_update_interval = trial.suggest_int("target_update_interval", 2, 100)
        train_freq = 1
        gradient_steps = 1
        exploration_initial_eps = trial.suggest_float(
            "exploration_initial_eps", 0.5, 1.0
        )
        exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.25)
        exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.0, 0.1)
        optimizer = trial.suggest_categorical(
            "optimizer", ["Adam", "AdamW", "RmsProp", "Sgd"]
        )
        loss_fn = trial.suggest_categorical(
            "loss_fn", ["MAE", "MSE", "RMSE", "Huber", "smooth_l1"]
        )

        model = DQN(
            net_arch=[(256, "relu"), (256, "relu")],
            learning_rate=learning_rate,
            last_activation="none",
            memory_size=buffer_size,
            min_memory_size=min_buffer_size,
            discount_factor=0.99,
            initial_epsilon=exploration_initial_eps,
            final_epsilon=exploration_final_eps,
            exploration_fraction=exploration_fraction,
            seed=seed,
            optimizer=optimizer,
            loss_fn=loss_fn,
        )
        model.prepare(env)

        env.reset(seed=seed + 1)
        eval_env.reset(seed=seed + 2)

        training_steps = 0
        for i, steps in enumerate(range(10_000, 300_000, 10_000)):
            results = model.train(
                env,
                eval_env,
                env.spec.reward_threshold,
                steps=steps,
                gradient_steps=gradient_steps,
                train_freq=train_freq,
                update_freq=target_update_interval,
                batch_size=batch_size,
                eval_freq=eval_freq,
                eval_for=eval_size,
                verbose=verbose,
            )
            training_steps += sum(results[1])
            if training_steps >= least_steps_number:
                reward, std = model.evaluate(eval_env, eval_size)
                return training_steps, reward
        reward, std = model.evaluate(eval_env, eval_size)
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
        n_jobs=-1,
        show_progress_bar=True,
    )
    print(study.best_params)
