import torch
import torch.nn as nn

import random
import numpy as np

import nasimemu.env_utils as env_utils


# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


# Experience Replay Buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Function to pad states dynamically
def pad_state(state, max_rows, max_cols):
    """Pad state with zeros if it's smaller than max_rows and max_cols"""
    current_rows, current_cols = state.shape
    padded_state = torch.zeros((max_rows, max_cols), device=state.device)
    padded_state[:current_rows, :current_cols] = state
    return padded_state

# Get list of valid actions at each step
def get_valid_actions(env, state):
    actions = env_utils.get_possible_actions(env, state)
    return actions if actions else [env.action_space.sample()]

# Function to convert PyTorch tensor to numpy.int64
def tensor_to_numpy_int64(tensor):
    return np.int64(tensor.item())