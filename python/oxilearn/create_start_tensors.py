from oxilearn import DQNAgent
import gymnasium as gym

env = gym.make('CartPole-v1')
seed = 42
agent = DQNAgent([(256, "relu"), (256, "relu")])
agent.prepare(env)
agent.save("./safetensors")


