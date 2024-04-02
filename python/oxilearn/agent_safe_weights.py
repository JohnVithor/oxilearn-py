from datetime import datetime
from oxilearn import DQNAgent
import gymnasium as gym

env = gym.make('CartPole-v1')
seed = 42
agent = DQNAgent([(128, "relu"), (128, "relu")])
agent.prepare(env)
agent.save("./safetensors")


