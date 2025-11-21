import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleQNetwork(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], n_actions: int):
        super(SimpleQNetwork, self).__init__()
        c, h, w = input_shape
        # a very small conv-net followed by linear layers
        # ensure h and w are divisible by 4 for the simple downsampling
        assert h % 4 == 0 and w % 4 == 0, "height and width must be divisible by 4"
        self.net = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (h // 4) * (w // 4), 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """Very small DQN agent skeleton.

    - input states are numpy arrays shaped (C,H,W) and in [0,255] or floats.
    - actions are indices into the provided action list.
    """

    def __init__(self, action_space: List[str], state_shape: Tuple[int, ...], lr: float = 1e-3,
                 gamma: float = 0.99, device=None):
        self.action_space = action_space
        self.n_actions = len(action_space)
        self.state_shape = state_shape
        self.gamma = gamma
        self.device = device if device is not None else default_device()

        self.policy_net = SimpleQNetwork(state_shape, self.n_actions).to(self.device)
        self.target_net = SimpleQNetwork(state_shape, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # epsilon for epsilon-greedy
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def select_action(self, state: np.ndarray) -> int:
        # state: (C,H,W) numpy
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        self.policy_net.eval()
        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0) / 255.0
            q = self.policy_net(t)
            action = int(q.argmax(dim=1).item())
        return action

    def optimize_step(self, batch, target_update=False):
        # batch: tuple of (states, actions, rewards, next_states, dones)
        states, actions, rewards, next_states, dones = batch
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device) / 255.0
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device) / 255.0
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1)[0]
            expected_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(q_values, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if target_update:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Epsilon decay should be called explicitly per environment step or episode.
        return float(loss.item())

    def action_to_string(self, action_idx: int) -> str:
        return self.action_space[action_idx]

    def decay_epsilon(self):
        """Decay epsilon, to be called per environment step or episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
