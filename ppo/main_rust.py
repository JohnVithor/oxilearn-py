import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import random

from model import Policy
from oxilearn import PPO


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

    agent = PPO()
    print(f"created")
    agent.prepare(env, eval_env)
    print(f"prepared")
    agent.train()
    print(f"Trained")

    results = agent.evaluate(10)
    print(f"Results: {results}")

    env.close()
