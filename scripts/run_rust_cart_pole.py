import sys
import gymnasium as gym
import os
import torch
import numpy as np
import random
from oxilearn import DQNAgent


def main(seed, save, verbose):

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1")

    agent = DQNAgent(
        net_arch=[(256, "relu"), (256, "relu")],
        learning_rate=0.0005,
        last_activation="none",
        memory_size=5_000,
        min_memory_size=1_000,
        discount_factor=0.99,
        initial_epsilon=1.0,
        final_epsilon=0.05,
        exploration_fraction=0.05,
        max_grad_norm=10.0,
        seed=seed,
        optimizer="Adam",
        loss_fn="MSE",
    )
    agent.prepare(env)

    env.reset(seed=seed + 1)
    eval_env.reset(seed=seed + 2)

    results = agent.train(
        env,
        eval_env,
        env.spec.reward_threshold,
        steps=100_000,
        gradient_steps=4,
        train_freq=2,
        update_freq=10,
        batch_size=64,
        eval_at=1000,
        eval_for=10,
        verbose=verbose,
    )
    training_steps = sum(results[1])

    if save:
        agent.save("./safetensors-rust")

    return training_steps, agent.evaluate(env, 10)


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
