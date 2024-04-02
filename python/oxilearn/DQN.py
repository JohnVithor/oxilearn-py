import sys
import os
import json
import time
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from datetime import datetime
from safetensors.torch import load_file

def main():
    train_steps = 50_000 if len(sys.argv) == 1 else int(sys.argv[1])
    output = {'train_steps': train_steps}

    env_id = "CartPole-v1"
    output['env'] = env_id

    path_info_data = './cartpole_data'
    eval_size = 10
    callback_freq = 1_000

    vec_env = make_vec_env(
        env_id, n_envs=1, vec_env_cls=DummyVecEnv, monitor_dir=f'{path_info_data}/single_training_info.log')
    
    eval_env = make_vec_env(
        env_id, n_envs=1, vec_env_cls=DummyVecEnv, monitor_dir=f'{path_info_data}/single_eval_info.log')


    if os.path.exists(path_info_data) and os.path.isfile(f'{path_info_data}/single_{env_id}_last_model.zip'):
        model = DQN.load(f'{path_info_data}/single_{env_id}_last_model.zip')   
        model.load_replay_buffer(f'{path_info_data}/single_{env_id}_last_replay.pkl')
        model.set_env(vec_env)
        print("model loaded")
    else:
        model = DQN(policy="MlpPolicy", env=vec_env, learning_rate=2.3e-3, batch_size=64,
                    buffer_size=100_000, learning_starts=1_000, gamma=0.99,
                    target_update_interval=10, train_freq=256, gradient_steps=128,
                    exploration_fraction=0.16, exploration_final_eps=0.04,
                    policy_kwargs={'net_arch': [128, 128]})
        if os.path.exists('./safetensors'):
            print("safetensors loaded")
            model.q_net.load_state_dict(load_file('/home/johnvithor/oxilearn/safetensors/policy_weights.safetensors'))
            model.q_net_target.load_state_dict(load_file('/home/johnvithor/oxilearn/safetensors/target_policy_weights.safetensors'))
    
    mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=eval_size)
    output['before_training'] = {
        'mean_reward': mean_reward,
        'std_reward': std_reward
    }

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=475, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, eval_freq=callback_freq, verbose=1)

    model.learn(total_timesteps=train_steps, callback=[eval_callback])

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=eval_size)
    output['after_training'] = {
        'mean_reward': mean_reward, 'std_reward': std_reward}

    print(json.dumps(output))


if __name__ == '__main__':
    start = time.time()
    print("start:", datetime.fromtimestamp(start).strftime("%Y/%m/%d %H:%M:%S"))
    main()
    end = time.time()
    print("end:", datetime.fromtimestamp(end).strftime("%Y/%m/%d %H:%M:%S"))
    print("elapsed:", end - start, 'seconds')

