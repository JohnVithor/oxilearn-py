import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import random

from model import Policy
from agent import PPO


if __name__ == "__main__":
    env_id = "CartPole-v1"
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))
    eval_env = gym.make(env_id)
    eval_env.reset(seed=seed)

    assert isinstance(
        env.action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    policy = Policy(
        np.array(env.observation_space.shape).prod(), env.action_space.n
    ).to(device)

    optimizer = optim.Adam(policy.parameters(), lr=0.0005)

    trainer = PPO(
        policy,
        optimizer,
        env,
        eval_env,
        num_steps=100,
        learning_rate=0.0005,
        device=device,
    )

    trainer.learn(100_000, seed)
    print(f"Training ended")

    results = trainer.evaluate(10)
    print(f"Results: {results}")

    env.close()
