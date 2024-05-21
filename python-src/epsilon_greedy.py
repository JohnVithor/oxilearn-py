import numpy as np
import torch
from typing import Callable, Union
from dataclasses import dataclass


class EpsilonUpdateStrategy:
    @dataclass
    class AdaptativeEpsilon:
        min_epsilon: float
        max_epsilon: float
        min_reward: float
        max_reward: float
        eps_range: float

    @dataclass
    class EpsilonCustomDecreasing:
        final_epsilon: float
        epsilon_decay: Callable[[float], float]

    @dataclass
    class EpsilonLinearTrainingDecreasing:
        start: float
        end: float
        end_fraction: float

    def update(
        self,
        strategy,
        current_epsilon: float,
        current_training_progress: float,
        epi_reward: float,
    ) -> float:
        if isinstance(strategy, EpsilonUpdateStrategy.AdaptativeEpsilon):
            if epi_reward < strategy.min_reward:
                return strategy.max_epsilon
            else:
                reward_range = strategy.max_reward - strategy.min_reward
                min_update = strategy.eps_range / reward_range
                new_eps = (strategy.max_reward - epi_reward) * min_update
                return max(new_eps, strategy.min_epsilon)
        elif isinstance(strategy, EpsilonUpdateStrategy.EpsilonCustomDecreasing):
            new_epsilon = strategy.epsilon_decay(current_epsilon)
            return max(new_epsilon, strategy.final_epsilon)
        elif isinstance(
            strategy, EpsilonUpdateStrategy.EpsilonLinearTrainingDecreasing
        ):
            if current_training_progress > strategy.end_fraction:
                return strategy.end
            else:
                return (
                    strategy.start
                    + current_training_progress
                    * (strategy.end - strategy.start)
                    / strategy.end_fraction
                )
        else:
            return current_epsilon


class EpsilonGreedy:
    def __init__(
        self,
        epsilon: float,
        seed: int,
        update_strategy: Union[
            None,
            EpsilonUpdateStrategy.AdaptativeEpsilon,
            EpsilonUpdateStrategy.EpsilonCustomDecreasing,
            EpsilonUpdateStrategy.EpsilonLinearTrainingDecreasing,
        ],
    ):
        self.initial_epsilon = epsilon
        self.current_epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.update_strategy = update_strategy

    @classmethod
    def default(cls):
        return cls(0.1, 42, None)

    def should_explore(self) -> bool:
        return (
            self.current_epsilon != 0.0
            and self.rng.uniform(0.0, 1.0) <= self.current_epsilon
        )

    def get_action(self, values: torch.Tensor) -> int:
        if self.should_explore():
            return self.rng.integers(0, values.size(0))
        else:
            return int(torch.argmax(values).item())

    def get_epsilon(self) -> float:
        return self.current_epsilon

    def reset(self):
        self.current_epsilon = self.initial_epsilon

    def update(self, current_training_progress: float, epi_reward: float):
        self.current_epsilon = EpsilonUpdateStrategy().update(
            self.update_strategy,
            self.current_epsilon,
            current_training_progress,
            epi_reward,
        )
