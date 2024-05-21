import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from experience_buffer import RandomExperienceBuffer
from epsilon_greedy import EpsilonGreedy
from typing import List, Callable, Tuple

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]


class PolicyGenerator(nn.Module):
    def __init__(
        self,
        net_arch: List[Tuple[int, ActivationFunction]],
        last_activation: ActivationFunction,
        input_dim: int,
        output_dim: int,
    ):
        super(PolicyGenerator, self).__init__()
        layers = []
        previous = input_dim
        for i, (neurons, activation) in enumerate(net_arch):
            layers.append(nn.Linear(previous, neurons))
            layers.append(activation)
            previous = neurons
        layers.append(nn.Linear(previous, output_dim))
        layers.append(last_activation)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def generate_policy(
    net_arch: List[Tuple[int, ActivationFunction]],
    last_activation: ActivationFunction,
    input_dim: int,
    output_dim: int,
) -> Callable[[str, torch.device], Tuple[nn.Module, torch.optim.Optimizer]]:
    def policy_generator(
        name: str, device: torch.device
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        model = PolicyGenerator(net_arch, last_activation, input_dim, output_dim).to(
            device
        )
        return model

    return policy_generator


# Funções de perda
def mae(values, expected_values):
    return F.l1_loss(values, expected_values, reduction="mean")


def mse(values, expected_values):
    return F.mse_loss(values, expected_values, reduction="mean")


def rmse(values, expected_values):
    return torch.sqrt(F.mse_loss(values, expected_values, reduction="mean"))


def huber(values, expected_values):
    return F.huber_loss(values, expected_values, reduction="mean", delta=1.35)


def smooth_l1(values, expected_values):
    return F.smooth_l1_loss(values, expected_values, reduction="mean", beta=1.0 / 9.0)


# Enum de otimizadores
class OptimizerEnum:
    def __init__(self, opt_type, **kwargs):
        self.opt_type = opt_type
        self.kwargs = kwargs

    def build(self, parameters, lr):
        if self.opt_type == "adam":
            return optim.Adam(parameters, lr=lr, **self.kwargs)
        elif self.opt_type == "sgd":
            return optim.SGD(parameters, lr=lr, **self.kwargs)
        elif self.opt_type == "rmsprop":
            return optim.RMSprop(parameters, lr=lr, **self.kwargs)
        elif self.opt_type == "adamw":
            return optim.AdamW(parameters, lr=lr, **self.kwargs)
        else:
            raise ValueError("Unknown optimizer type")


# Classe do agente Double Deep Q-Learning
class DoubleDeepAgent:
    def __init__(
        self,
        action_selector,
        mem_replay,
        generate_policy,
        opt,
        loss_fn,
        learning_rate,
        discount_factor,
        max_grad_norm,
        device,
    ):
        self.policy_net = generate_policy("q_net", device)
        self.target_net = generate_policy("q_net", device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = opt(self.policy_net.parameters(), learning_rate)
        self.loss_fn = loss_fn
        self.action_selection = action_selector
        self.memory = mem_replay
        self.max_grad_norm = max_grad_norm
        self.discount_factor = discount_factor
        self.device = device

    def get_action(self, state):
        with torch.no_grad():
            values = self.policy_net(state.to(self.device))
        return self.action_selection.get_action(values)

    def get_best_action(self, state):
        with torch.no_grad():
            values = self.policy_net(state.to(self.device))
        return values.argmax(dim=0).item()

    def add_transition(self, curr_state, curr_action, reward, done, next_state):
        self.memory.add(curr_state, curr_action, reward, done, next_state)

    def update_networks(self):
        return self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_batch(self, size):
        return self.memory.sample_batch(size)

    def batch_qvalues(self, b_states, b_actions):
        return self.policy_net(b_states).gather(1, b_actions.long()).float()

    def batch_expected_values(self, b_state_, b_reward, b_done):
        with torch.no_grad():
            best_target_qvalues = self.target_net(b_state_).max(dim=1, keepdim=True)[0]
        return (
            b_reward + self.discount_factor * (1 - b_done.float()) * best_target_qvalues
        ).float()

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def update(self, gradient_steps, batch_size):
        if not self.memory.ready():
            return None
        values = []
        for _ in range(gradient_steps):
            b_state, b_action, b_reward, b_done, b_state_ = self.get_batch(batch_size)
            policy_qvalues = self.batch_qvalues(b_state, b_action)
            expected_values = self.batch_expected_values(b_state_, b_reward, b_done)
            loss = self.loss_fn(policy_qvalues, expected_values)
            self.optimize(loss)
            values.append(expected_values.mean().item())
        return sum(values) / len(values)

    def action_selection_update(self, current_training_progress, epi_reward):
        self.action_selection.update(current_training_progress, epi_reward)

    def get_epsilon(self):
        return self.action_selection.get_epsilon()

    def reset(self):
        self.action_selection.reset()

    def save_net(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(
            self.policy_vs.state_dict(), os.path.join(path, "policy_weights.pth")
        )
        torch.save(
            self.target_policy_vs.state_dict(),
            os.path.join(path, "target_policy_weights.pth"),
        )

    def load_net(self, path):
        self.policy_vs.load_state_dict(
            torch.load(os.path.join(path, "policy_weights.pth"))
        )
        self.target_policy_vs.load_state_dict(
            torch.load(os.path.join(path, "target_policy_weights.pth"))
        )
