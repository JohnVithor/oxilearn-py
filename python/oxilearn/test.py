from oxilearn import DQNAgent
import gymnasium as gym

# env = gym.make('MountainCar-v0')
env = gym.make('CartPole-v1')
(obs, info) = env.reset(seed=0)
# (obs, reward, terminated, truncated, info) = env.step(0)
# print((obs, reward, terminated, truncated, info))

#####################################

agent = DQNAgent([(128, "relu")]).train(env)

r = agent
print(r)

