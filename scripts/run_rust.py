import sys
from datetime import datetime
import gymnasium as gym
import os
import torch 
import numpy as np
import random
from oxilearn import DQNAgent

def main(seed, save):

    env = gym.make('CartPole-v1')
    agent = DQNAgent([(256, "relu"), (256, "relu")])
    agent.prepare(env)
    agent.load("./safetensors")
    env.reset(seed=seed)

    start = datetime.now()
    agent.train(env, 500.0, steps=1_000_000, verbose=0)
    end = datetime.now()

    print(f"{end-start}", end=" ")
    
    if save:
        agent.save("./safetensors-rust")

    return agent.evaluate(env, 10)

if __name__ == '__main__':
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    save = bool(sys.argv[2]) if len(sys.argv) > 2 else False

    os.environ['PYTHONASHSEED'] = f'{seed}' 
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)

    reward_info = main(seed, save)
    print(f"{reward_info}")


