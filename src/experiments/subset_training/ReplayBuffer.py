import random
from collections import deque
from typing import Deque

import numpy as np


class ReplayBuffer:
    """Simple fixed-size replay buffer storing transitions as tuples.

    Transitions: (state, action, reward, next_state, done)
    States are numpy arrays.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: Deque = deque(maxlen=capacity)

    def add(self, state, action: int, reward: float, next_state, done: bool):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Convert to numpy arrays
        states_arr = np.stack(states)
        next_states_arr = np.stack(next_states)
        actions_arr = np.array(actions, dtype=np.int64)
        rewards_arr = np.array(rewards, dtype=np.float32)
        dones_arr = np.array(dones, dtype=np.float32)
        return states_arr, actions_arr, rewards_arr, next_states_arr, dones_arr

    def __len__(self):
        return len(self.buffer)
