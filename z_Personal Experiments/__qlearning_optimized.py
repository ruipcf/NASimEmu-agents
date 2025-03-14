import torch
import torch.nn as nn
import torch.optim as optim

import json
import time
import itertools
import random
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

import gymnasium as gym

from __utils import *


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameter grid with extended parameters for episodes, steps per episode, and network architecture
param_grid = {
    'LEARNING_RATE': [1e-3, 1e-4],
    'GAMMA': [0.99, 0.95],
    'EPSILON_DECAY': [0.999, 0.995],
    'MIN_EPSILON': [0.1, 0.01],
    'BATCH_SIZE': [64, 128],
    'MEMORY_SIZE': [10000, 20000],
    'TARGET_UPDATE': [10, 20],
    'EPISODES': [500, 1000],
    'STEPS_PER_EPISODE': [200, 500],
    # 'Q_ARCHITECTURE': [
    #     {'layers': [128, 128], 'layer_type': 'Linear'},
    #     {'layers': [256, 128, 64], 'layer_type': 'Linear'},
    #     {'layers': [128, 128], 'layer_type': 'Conv2d'}
    # ]
}

# Function to run the experiment with different hyperparameters
def run_experiment(hyperparameters):

    EPSILON =  1.0

    LEARNING_RATE = hyperparameters['LEARNING_RATE']
    GAMMA = hyperparameters['GAMMA']
    EPSILON_DECAY = hyperparameters['EPSILON_DECAY']
    MIN_EPSILON = hyperparameters['MIN_EPSILON']
    BATCH_SIZE = hyperparameters['BATCH_SIZE']
    MEMORY_SIZE = hyperparameters['MEMORY_SIZE']
    TARGET_UPDATE = hyperparameters['TARGET_UPDATE']
    EPISODES = hyperparameters['EPISODES']
    STEPS_PER_EPISODE = hyperparameters['STEPS_PER_EPISODE']
    # Q_ARCHITECTURE = hyperparameters['Q_ARCHITECTURE']

    # Initialize environment
    env = gym.make('NASimEmu-v0', emulate=False, 
                   scenario_name='NASimEmu/scenarios/md_entry_dmz_one_subnet.v2.yaml:NASimEmu/scenarios/md_entry_dmz_one_subnet.v2.yaml')
    env = env.unwrapped

    # Dynamically determine initial STATE shape
    initial_state, _ = env.reset()
    MAX_STATE_ROWS, MAX_STATE_COLS = initial_state.shape
    INPUT_DIM = MAX_STATE_ROWS * MAX_STATE_COLS
    ACTION_DIM = len(env.unwrapped.action_list)

    # Create Q-network based on the specified architecture
    q_net = QNetwork(INPUT_DIM, ACTION_DIM).to(device)
    target_net = QNetwork(INPUT_DIM, ACTION_DIM).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    reward_list = []
    start_time = time.time()

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

                # print(f"⚠️ Reinitialized Q-network with new input size: {MAX_STATE_ROWS}x{MAX_STATE_COLS}, action size: {ACTION_DIM}")

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
                
                # print(f"⚠️ Reinitialized Q-network with new input size: {MAX_STATE_ROWS}x{MAX_STATE_COLS}, action size: {ACTION_DIM}")

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

    end_time = time.time()

    # Save logs to JSON
    experiment_time = end_time - start_time
    logs = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": hyperparameters,
        "total_training_time": experiment_time,
        "reward_list": reward_list,
        "final_reward": reward_list[-1],
        "total_rewards": sum(reward_list)
    }

    file_name = f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(file_name, 'w') as f:
        json.dump(logs, f, indent=4)

    env.close()

# Grid search function to test all combinations
def grid_search(param_grid):
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    for params in tqdm(combinations):
        run_experiment(params)

# Start grid search
grid_search(param_grid)
