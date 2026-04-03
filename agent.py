"""
agent.py — Naive Deep Q-Learning agent for CartPole.

Stabilization techniques used:
  - Huber loss (less sensitive to outliers than MSE)
  - Gradient clipping
  - Target values clamped to reasonable range
  - Layer initialization (orthogonal) for stable early training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        in_size = state_dim
        for h in hidden_sizes:
            linear = nn.Linear(in_size, h)
            # Orthogonal init: helps prevent vanishing/exploding gradients early on
            nn.init.orthogonal_(linear.weight, gain=np.sqrt(2))
            nn.init.constant_(linear.bias, 0.0)
            layers += [linear, nn.ReLU()]
            in_size = h
        out = nn.Linear(in_size, action_dim)
        nn.init.orthogonal_(out.weight, gain=0.01)
        nn.init.constant_(out.bias, 0.0)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Naive DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [64, 64],
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 100_000,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        self.q_net = QNetwork(state_dim, action_dim, hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr, eps=1e-5)

        # Huber loss: behaves like MSE for small errors, L1 for large ones
        # This prevents enormous loss values when Q-values are temporarily off
        self.loss_fn = nn.HuberLoss(delta=1.0)

        self.step_count = 0

    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.step_count / self.epsilon_decay_steps)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax(dim=1).item()

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float:
        """
        DQN loss (Task 1.1):
          TD target:  y = r + gamma * max_a' Q(s', a')   (0 if terminal)
          Loss:       L = HuberLoss(Q(s,a), y)
        """
        s  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        ns = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        a  = torch.LongTensor([action]).to(self.device)
        r  = torch.FloatTensor([reward]).to(self.device)
        d  = torch.FloatTensor([float(done)]).to(self.device)

        q_current = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.q_net(ns).max(dim=1).values
            target = r + self.gamma * q_next * (1.0 - d)
            # Clamp target to CartPole's realistic return range [0, 500]
            target = target.clamp(0.0, 500.0)

        loss = self.loss_fn(q_current, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.step_count += 1
        return loss.item()

    def increment_step(self):
        self.step_count += 1

    def get_state_dict(self):
        return self.q_net.state_dict()

    def load_state_dict(self, sd):
        self.q_net.load_state_dict(sd)
