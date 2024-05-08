import sys
import torch
import numpy as np
import random
import os

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

import optuna


def create_objective(env_name, seed, verbose):

    def objective(trial) -> float:

        eval_size = 10
        callback_freq = 1_000

        vec_env = make_vec_env(env_name, seed=seed, n_envs=1, vec_env_cls=DummyVecEnv)
        eval_env = make_vec_env(
            env_name, seed=seed + 1, n_envs=1, vec_env_cls=DummyVecEnv
        )

        learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.1)
        batch_size = trial.suggest_int("batch_size", 16, 256)
        buffer_size = trial.suggest_int("buffer_size", 1_000, 100_000)
        learning_starts = trial.suggest_int("learning_starts", 0, 10_000)
        gamma = trial.suggest_float("gamma", 0.9, 0.99)
        target_update_interval = trial.suggest_int("target_update_interval", 1, 256)
        train_freq = trial.suggest_int("train_freq", 1, 256)
        gradient_steps = trial.suggest_int("gradient_steps", -1, 256)
        exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.25)
        exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.0, 0.1)

        model = DQN(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            gamma=gamma,
            target_update_interval=target_update_interval,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            exploration_initial_eps=1.00,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            policy_kwargs={"net_arch": [256, 256]},
            seed=seed + 2,
        )

        callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=vec_env.get_attr("spec")[0].reward_threshold,
            verbose=verbose,
        )
        eval_callback = EvalCallback(
            eval_env,
            callback_on_new_best=callback_on_best,
            eval_freq=callback_freq,
            verbose=verbose,
        )

        model.learn(total_timesteps=100_000, callback=[eval_callback])

        reward, std = evaluate_policy(model, eval_env, n_eval_episodes=eval_size)

        return reward

    return objective


if __name__ == "__main__":
    env_name = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 1 else 0
    verbose = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    os.environ["PYTHONASHSEED"] = f"{seed}"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    study_name = f"{env_name}"
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(create_objective(env_name, seed, verbose), n_trials=100)
    print(study.best_params)
