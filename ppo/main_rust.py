import gymnasium as gym
import torch
import torch.optim as optim
import numpy as np
import random

from model import Policy
from oxilearnpy import PPO


if __name__ == "__main__":
    env_id = "CartPole-v1"
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    env = gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))
    env.reset(seed=seed)
    eval_env = gym.make(env_id)
    eval_env.reset(seed=seed)

    assert isinstance(
        env.action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    agent = PPO(seed)
    obs_shape = env.observation_space.shape
    n_action = env.action_space.n
    agent.prepare(obs_shape[0], n_action)
    agent.train(env, eval_env)

    results = agent.evaluate(eval_env, 10)
    print(f"Results: {results}")

    env.close()
