from datetime import datetime
from oxilearn import DQNAgent
import gymnasium as gym

env = gym.make('CartPole-v1')
seed = 42
agent = DQNAgent([(128, "relu"), (128, "relu")])
agent.prepare(env)
agent.load("./safetensors")

start = datetime.now()
_ = env.reset(seed=seed)
max_reward = agent.train(env, 500.0, verbose=1)
end = datetime.now()

execution_time = end-start
print(f" {max_reward}, {execution_time}")



