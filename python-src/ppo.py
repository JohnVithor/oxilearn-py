import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PPOAgent:
    def __init__(
        self,
        action_selector,
        mem_replay,
        generate_policy,
        generate_value,
        opt,
        loss_fn,
        learning_rate,
        discount_factor,
        epsilon_clip,
        max_grad_norm,
        device,
    ):
        self.policy_net = generate_policy(device)
        self.value_net = generate_value(device)
        self.policy_net.double()
        self.value_net.double()

        self.optimizer_policy = opt(self.policy_net.parameters(), learning_rate)
        self.optimizer_value = opt(self.value_net.parameters(), learning_rate)
        self.loss_fn = loss_fn
        self.action_selection = action_selector
        self.memory = mem_replay
        self.max_grad_norm = max_grad_norm
        self.discount_factor = discount_factor
        self.epsilon_clip = epsilon_clip
        self.device = device

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.double).to(self.device)
        with torch.no_grad():
            probs = self.policy_net(state)
            dist = Categorical(probs)
            action = dist.sample()
        return action.item(), dist.log_prob(action)

    def add_transition(
        self, curr_state, curr_action, reward, done, next_state, log_prob
    ):
        self.memory.add(curr_state, curr_action, reward, done, next_state, log_prob)

    def compute_gae(self, rewards, dones, values, next_values, gamma, lam):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, gradient_steps, batch_size):
        if not self.memory.ready():
            return None
        states, actions, rewards, dones, next_states, log_probs = (
            self.memory.sample_batch(batch_size)
        )

        states = torch.tensor(states, dtype=torch.double).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.double).to(self.device)
        dones = torch.tensor(dones, dtype=torch.double).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.double).to(self.device)
        log_probs = torch.tensor(log_probs, dtype=torch.double).to(self.device)

        with torch.no_grad():
            values = self.value_net(states)
            next_values = self.value_net(next_states)
            advantages = self.compute_gae(
                rewards, dones, values, next_values, self.discount_factor, 0.95
            )
            returns = advantages + values

        advantages = torch.tensor(advantages, dtype=torch.double).to(self.device)
        returns = torch.tensor(returns, dtype=torch.double).to(self.device)

        for _ in range(gradient_steps):
            probs = self.policy_net(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            ratio = (new_log_probs - log_probs).exp()
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip)
                * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), self.max_grad_norm
            )
            self.optimizer_policy.step()

        for _ in range(gradient_steps):
            value_loss = self.loss_fn(self.value_net(states).squeeze(), returns)

            self.optimizer_value.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.value_net.parameters(), self.max_grad_norm
            )
            self.optimizer_value.step()

        return policy_loss.item(), value_loss.item()

    def save_net(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"{path}/policy_weights.pth")
        torch.save(self.value_net.state_dict(), f"{path}/value_weights.pth")

    def load_net(self, path):
        self.policy_net.load_state_dict(torch.load(f"{path}/policy_weights.pth"))
        self.value_net.load_state_dict(torch.load(f"{path}/value_weights.pth"))
