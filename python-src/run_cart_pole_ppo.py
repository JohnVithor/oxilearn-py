import gymnasium as gym
import sys
import torch
from torch import nn
from dqn import generate_policy
from ppo import PPOAgent
from epsilon_greedy import EpsilonGreedy, EpsilonUpdateStrategy
from experience_buffer import RandomExperienceBuffer
from trainer import Trainer, TrainResults
from cart_pole import CartPoleEnv


def main():
    args = sys.argv
    seed = int(args[1])
    verbose = int(args[2])
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    # train_env = gym.make("CartPole-v1")
    train_env = CartPoleEnv()
    train_env.reset(seed=seed)
    # eval_env = gym.make("CartPole-v1")
    eval_env = CartPoleEnv()
    eval_env.reset(seed=seed + 1)

    update_strategy = EpsilonUpdateStrategy.EpsilonLinearTrainingDecreasing(
        start=1.0, end=0.05, end_fraction=0.2
    )
    action_selector = EpsilonGreedy(1.0, seed + 2, update_strategy)

    mem_replay = RandomExperienceBuffer(10_000, 4, 1_000, seed + 3, False, device)

    policy = generate_policy(
        [(256, nn.ReLU()), (256, nn.ReLU())],
        nn.Identity(),
        4,
        2,
    )

    optimizer = torch.optim.Adam
    loss_fn = nn.HuberLoss()

    model = PPOAgent(
        action_selector,
        mem_replay,
        policy,
        policy,
        optimizer,
        loss_fn,
        0.003,
        0.99,
        1.0,
        1.0,
        device,
    )
    # model.save_net("./safetensors-python/cart_pole")

    trainer = Trainer(train_env, eval_env)
    trainer.early_stop = lambda reward: reward >= 475.0

    training_results = trainer.train_by_steps(
        model, 50_000, 175, 200, 128, 10, 1000, 10, verbose
    )
    training_steps = sum(training_results[1])

    evaluation_results = trainer.evaluate(model, 1)
    rewards = evaluation_results[0]
    reward_avg = sum(rewards) / len(rewards)
    variance = sum((reward_avg - value) ** 2 for value in rewards) / len(rewards)
    std = variance**0.5

    # model.save_net("./safetensors-python/cart_pole_after_training")

    print(f"python,{seed},{training_steps},{reward_avg},{std}")


if __name__ == "__main__":
    main()
