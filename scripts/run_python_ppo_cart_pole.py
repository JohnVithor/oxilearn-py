import sys
import torch
import numpy as np
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from safetensors.torch import load_file, save_file


def main(seed, save, verbose):
    eval_size = 10
    callback_freq = 1_000

    vec_env = make_vec_env("CartPole-v1", seed=seed, n_envs=1, vec_env_cls=DummyVecEnv)
    eval_env = make_vec_env("CartPole-v1", seed=seed, n_envs=1, vec_env_cls=DummyVecEnv)

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=1e-3,
        gamma=0.9,
        n_steps=1024,
        n_epochs=10,
        gae_lambda=0.95,
        ent_coef=0.0,
        clip_range=0.2,
        seed=seed,
    )

    reward_threshold = 475.0
    callback_on_best = StopTrainingOnRewardThreshold(
        reward_threshold=reward_threshold, verbose=verbose
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        eval_freq=callback_freq,
        verbose=verbose,
    )

    model.learn(total_timesteps=1_000_000, callback=[eval_callback])

    if save:
        os.makedirs("./safetensors-python", exist_ok=True)
        save_file(
            model.policy.mlp_extractor.policy_net.state_dict(),
            "./safetensors-python/policy_weights.safetensors",
        )
        save_file(
            model.policy.mlp_extractor.value_net.state_dict(),
            "./safetensors-python/value_weights.safetensors",
        )
        save_file(model.policy.state_dict(), "./safetensors-python/p.safetensors")

    return model.num_timesteps, evaluate_policy(
        model, eval_env, n_eval_episodes=eval_size, render=True
    )


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    save = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else False
    verbose = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    os.environ["PYTHONASHSEED"] = f"{seed}"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    training_steps, (reward, std) = main(seed, save, verbose)
    print(f"python,{seed},{training_steps},{reward},{std}")
