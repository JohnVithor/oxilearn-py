import sys
import torch 
import numpy as np
import random
import os
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from datetime import datetime
from safetensors.torch import load_file, save_file


def main(seed, save):
    eval_size = 10
    callback_freq = 1_000

    vec_env = make_vec_env("CartPole-v1", seed=seed, n_envs=1, vec_env_cls=DummyVecEnv)
    eval_env = make_vec_env("CartPole-v1", seed=seed, n_envs=1, vec_env_cls=DummyVecEnv)

    model = DQN(policy="MlpPolicy", env=vec_env, learning_rate=2.3e-3, batch_size=64,
                buffer_size=100_000, learning_starts=1_000, gamma=0.99,
                target_update_interval=10, train_freq=256, gradient_steps=128,
                exploration_initial_eps=1.00, exploration_fraction=0.16, exploration_final_eps=0.04,
                policy_kwargs={'net_arch': [256, 256]})
    if os.path.exists('./safetensors'):
        model.q_net.load_state_dict(load_file('./safetensors/policy_weights.safetensors'))
        model.q_net_target.load_state_dict(load_file('./safetensors/target_policy_weights.safetensors'))
    
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=500, verbose=0)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, eval_freq=callback_freq, verbose=0)

    start = datetime.now()
    model.learn(total_timesteps=1_000_000, callback=[eval_callback])
    end = datetime.now()

    print(f"{end-start}", end=" ")

    if save:
        os.mkdir('./safetensors-python') 
        save_file(model.q_net.state_dict(), './safetensors-python/policy_weights.safetensors')
        save_file(model.q_net_target.state_dict(), './safetensors-python/target_policy_weights.safetensors')

    return evaluate_policy(model, eval_env, n_eval_episodes=eval_size)

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

