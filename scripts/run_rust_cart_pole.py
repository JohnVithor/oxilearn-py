import sys
import gymnasium as gym
import os
import torch
import numpy as np
import random
from oxilearn import DQN


def main(seed, save, verbose):

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    model = DQN(
        net_arch=[(256, "relu"), (256, "relu")],
        learning_rate=0.01,
        last_activation="none",
        memory_size=10_000,
        min_memory_size=1_000,
        discount_factor=0.99,
        initial_epsilon=0.5,
        final_epsilon=0.05,
        exploration_fraction=0.2,
        max_grad_norm=10.0,
        seed=seed,
        optimizer="Adam",
        loss_fn="Huber",
    )

    env.reset(seed=seed + 1)
    eval_env.reset(seed=seed + 2)

    results = model.train(
        env,
        eval_env,
        env.spec.reward_threshold,
        steps=50_000,
        gradient_steps=200,
        train_freq=100,
        update_freq=2,
        batch_size=256,
        eval_freq=1_000,
        eval_for=10,
        verbose=verbose,
    )
    training_steps = sum(results[1])

    if save:
        model.save("./safetensors-rust")

    return training_steps, model.evaluate(env, 10)


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    save = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else False
    verbose = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    os.environ["PYTHONASHSEED"] = f"{seed}"
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    training_steps, (reward, std) = main(seed, save, verbose)
    print(f"rust,{seed},{training_steps},{reward},{std}")
