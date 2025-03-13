import torch
import torch.nn as nn
import torch.optim as optim

import random
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import nasimemu.env_utils as env_utils
from nasimemu.nasim.envs.render import Viewer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LEARNING_RATE = 1e-3
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999 #0.995
MIN_EPSILON = 0.1 #0.01
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
EPISODES = 500
STEPS_PER_EPISODE = 200

# Initialize NASimEmu environment
env = gym.make('NASimEmu-v0', emulate=False, 
    scenario_name='NASimEmu/scenarios/md_entry_dmz_one_subnet.v2.yaml:NASimEmu/scenarios/md_entry_dmz_one_subnet.v2.yaml')

env = env.unwrapped

# Dynamically determine initial STATE shape
initial_state, _ = env.reset()
MAX_STATE_ROWS, MAX_STATE_COLS = initial_state.shape
INPUT_DIM = MAX_STATE_ROWS * MAX_STATE_COLS
ACTION_DIM = len(env.unwrapped.action_list)

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

# Initialize Q-network and target network
q_net = QNetwork(INPUT_DIM, ACTION_DIM).to(device)
target_net = QNetwork(INPUT_DIM, ACTION_DIM).to(device)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_SIZE)

# Training loop
reward_list = []

for episode in range(EPISODES):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device)
    total_reward = 0

    for step in range(STEPS_PER_EPISODE):
        valid_actions = get_valid_actions(env, state)

        # Reinitialize the network if the state space expands
        if state.shape[0] > MAX_STATE_ROWS or state.shape[1] > MAX_STATE_COLS or len(env.unwrapped.action_list) != ACTION_DIM:
            MAX_STATE_ROWS = max(MAX_STATE_ROWS, state.shape[0])
            MAX_STATE_COLS = max(MAX_STATE_COLS, state.shape[1])
            ACTION_DIM = len(env.unwrapped.action_list)

            INPUT_DIM = MAX_STATE_ROWS * MAX_STATE_COLS
            q_net = QNetwork(INPUT_DIM, ACTION_DIM).to(device)
            target_net = QNetwork(INPUT_DIM, ACTION_DIM).to(device)
            target_net.load_state_dict(q_net.state_dict())

            optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
            memory = ReplayMemory(MEMORY_SIZE)

            print(f"⚠️ Reinitialized Q-network with new input size: {MAX_STATE_ROWS}x{MAX_STATE_COLS}, action size: {ACTION_DIM}")

        state = pad_state(state, MAX_STATE_ROWS, MAX_STATE_COLS)

        # Select action using epsilon-greedy policy
        if random.random() < EPSILON:
            action = random.choice(valid_actions)
        else:
            with torch.no_grad():
                q_values = q_net(state.view(1, -1))  # Flatten before passing
                action_idx = torch.argmax(q_values).item()
                action = valid_actions[action_idx]

        # Take action
        (subnet_tensor, host_tensor), action_id = action
        subnet = tensor_to_numpy_int64(subnet_tensor)
        host = tensor_to_numpy_int64(host_tensor)
        converted_action = ((subnet, host), action_id)
        
        # action_debug_mode = env._translate_action(converted_action)
        # action_debug_mode = {
        #     'type' : type(action_debug_mode).__name__,
        #     'name' : action_debug_mode.name,
        #     'prob': action_debug_mode.prob,
        #     'cost' : action_debug_mode.cost,
        # }

        network_debug_mode = env.env.network
        from nasimemu.nasim.envs.render import Viewer
        Viewer = Viewer(network_debug_mode)
        state_np = state.cpu().numpy()
        state_debug_mode = env.env.current_state
        Viewer.render_graph(state_debug_mode, show=True)

        next_state, reward, done, _, _ = env.step(converted_action)

        # Reinitialize the network if the state space expands
        if next_state.shape[0] > MAX_STATE_ROWS or next_state.shape[1] > MAX_STATE_COLS or len(env.unwrapped.action_list) != ACTION_DIM:
            MAX_STATE_ROWS = max(MAX_STATE_ROWS, next_state.shape[0])
            MAX_STATE_COLS = max(MAX_STATE_COLS, next_state.shape[1])
            ACTION_DIM = len(env.unwrapped.action_list)

            INPUT_DIM = MAX_STATE_ROWS * MAX_STATE_COLS
            q_net = QNetwork(INPUT_DIM, ACTION_DIM).to(device)
            target_net = QNetwork(INPUT_DIM, ACTION_DIM).to(device)
            target_net.load_state_dict(q_net.state_dict())

            optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
            memory = ReplayMemory(MEMORY_SIZE)

            print(f"⚠️ Reinitialized Q-network with new input size: {MAX_STATE_ROWS}x{MAX_STATE_COLS}, action size: {ACTION_DIM}")

        total_reward += reward
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        memory.push((state, action_id, reward, next_state, done))
        state = next_state

        # Train Q-network
        if len(memory) > BATCH_SIZE:

            transitions = memory.sample(BATCH_SIZE)
            batch = list(zip(*transitions))

            states = torch.stack([pad_state(s, MAX_STATE_ROWS, MAX_STATE_COLS) for s in batch[0]])
            flattened_states = states.view(states.shape[0], -1)

            actions = torch.tensor(batch[1], dtype=torch.int64, device=device).view(-1, 1)
            rewards = torch.tensor(batch[2], dtype=torch.float32, device=device).view(-1, 1)
            try:
                next_states = torch.stack([pad_state(s, MAX_STATE_ROWS, MAX_STATE_COLS) for s in batch[3]])
            except Exception as e:
                next_states = torch.stack([pad_state(s, MAX_STATE_ROWS, MAX_STATE_COLS) for s in batch[3]])
                pass
            dones = torch.tensor(batch[4], dtype=torch.float32, device=device).view(-1, 1)

            q_values = q_net(flattened_states).gather(1, actions)

            with torch.no_grad():
                flattened_next_states = next_states.view(next_states.shape[0], -1)
                q_next = target_net(flattened_next_states)
                max_next_q_values = q_next.max(dim=1, keepdim=True)[0]
                target_q_values = rewards + GAMMA * max_next_q_values * (1 - dones)

            loss = nn.functional.mse_loss(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(q_net.state_dict())

    reward_list.append(total_reward)
    print(f"Episode {episode + 1}/{EPISODES}, Reward: {total_reward:.3f}, Epsilon: {EPSILON:.3f}")

env.close()

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(reward_list, label="Rewards per episode", color="blue")
ax.set_xlabel("Episodes")
ax.set_ylabel("Rewards")
ax.legend()
fig.show()

