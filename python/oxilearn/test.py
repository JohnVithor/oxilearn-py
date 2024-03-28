from datetime import datetime
import numpy as np
from oxilearn import DQNAgent
import gymnasium as gym

env = gym.make('CartPole-v1')
seeds = [seed for seed in range(0,1)]
times = []
for seed in seeds:
    print(f"on seed {seed}")
    start = datetime.now()
    _ = env.reset(seed=seed)
    max_reward = DQNAgent([(128, "relu")]).train(env, 500.0)
    end = datetime.now()
    execution_time = end-start
    print(f"end seed {seed}: {max_reward}, {execution_time}")
    times.append(execution_time)

print([f"{time}" for time in times])
print(np.mean(times))

