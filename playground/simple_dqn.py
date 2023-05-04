import torch
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from torchinfo import summary
from tqdm import tqdm
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from collections import deque

import argparse

# IMPLEMENTATION OF DQN

# Deep Q-Network approximates a state-value function in a Q-Learning framework with a neural network.
# It's similare to Q-Learning with some main differences:

# 1) Each step update is, potentially, used in many weights updates, which allow for greater data efficency.
# 2) Learning directly from consecutive samples is inefficient, due to the strong correlations between the samples; randoming the samples breaks these correlations and therefore
#   reduces the variance of the updates
# 3) When learning on-policy the current parameter determine the next data sample that the parameters are trained on. By using experience replay the behavior distribution is averaged 
#   over many of its previous states, smoothing out learning and avoid oscillations or diverrgence in the parameters.
# NOTE: when learning by experience replay. it is necessary to learn off-policy. (that's why of Q-Learning)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Replay Buffers are a central part of off-policy RL algorithms.

# A replay buffer is a data structure that stores experiences of the agent as it interacts with the environment.
# Each experience is typically represented as a tuple containing the current state, the action taken, the resulting reward, and the next state. 
# The replay buffer can be thought of as a dataset of experiences that the agent can use to learn from.

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    # Add a new experience to the buffer.
    def add(self, experience):
        self.buffer.append(experience)

    # Samples a batch of experiences from the buffer of size batch_size and returns the experiences.
    def sample(self, batch_size):
        batch = np.random.choice(self.buffer, size=batch_size, replace=False)
        return batch
    
    # This method returns the number of experiences currently stored in the buffer.
    def __len__(self):
        return len(self.buffer)
        


# BUNCH OF TRAINING HYPERPARAMETERS
batch_size = 32
gamma = 0.99
eps_start = 0.9
eps_end = 0.05
eps_decay = 1000
tau = 0.005
learning_rate = 1e-4

# Create game environment
env = gym.make('CartPole-v1', render_mode = None)

# Replica of the model presented in the paper. Is a standard CNN.
class DQNModel(torch.nn.Module):
    def __init__(self, obs_size : int, action_size : int) -> None:
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=obs_size, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=action_size)
        )

    def forward(self, x):
        return self.net(x)
    

# eps scheduling.
def compute_eps(current_step, min, max, num_steps):
    alpha = current_step / num_steps
    alpha = np.clip(alpha, 0.0, 1.0)

    return min * (1 - alpha) + max * alpha


# init replay buffer
replay_buffer = ReplayBuffer(max_size=10000)

obs, info = env.reset()

# init action-value Q
obs_dim = len(obs) # after image processing we will have 1 color channel.
action_dim = env.action_space.n

policy_model = DQNModel(obs_dim, action_size=action_dim).to(device)
target_model = DQNModel(obs_dim, action_size=action_dim).to(device)

target_model.load_state_dict(policy_model.state_dict())

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate, amsgrad=True)

print(policy_model)


