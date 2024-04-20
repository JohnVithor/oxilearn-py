import sys
import os
from oxilearn import DQNAgent
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from safetensors.torch import load_file, save_file
from stable_baselines3 import DQN
import torch 
import numpy as np
import random

def rust(seed, path):
    env = gym.make('CartPole-v1')
    agent = DQNAgent([(256, "relu"), (256, "relu")], seed=seed)
    agent.prepare(env)
    agent.save(f"{path}")

def python(seed, path):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    vec_env = make_vec_env("CartPole-v1", seed=seed, n_envs=1, vec_env_cls=DummyVecEnv)
    model = DQN(policy="MlpPolicy", env=vec_env, learning_rate=2.3e-3, batch_size=64,
            buffer_size=100_000, learning_starts=1_000, gamma=0.99,
            target_update_interval=10, train_freq=256, gradient_steps=128,
            exploration_initial_eps=1.00, exploration_fraction=0.16, exploration_final_eps=0.04,
            policy_kwargs={'net_arch': [256, 256]})
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')
    save_file(model.q_net.state_dict(), f'{path}/policy_weights.safetensors')
    save_file(model.q_net_target.state_dict(), f'{path}/target_policy_weights.safetensors')

if __name__ == '__main__':
    mode = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    path = int(sys.argv[3]) if len(sys.argv) > 3 else "./safetensors"
    if mode == 0 :
        print("rust mode")
        rust(seed, path)
    else:
        print("python mode")
        python(seed, path)