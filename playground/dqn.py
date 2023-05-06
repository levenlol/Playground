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

# IMPLEMENTATION OF: Playing Atari with Deep Reinforcement Learning
# https://arxiv.org/pdf/1312.5602.pdf and https://arxiv.org/pdf/1711.07478.pdf and https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

# Parse commandline for game name and other infos.
def parse_args():
    parser = argparse.ArgumentParser(description='Parse command line arguments.')
    parser.add_argument('--game', type=str, default='Cartpole', help='Name of the game to play')
    parser.add_argument('--training', action='store_true', help='Enable training mode')
    args = parser.parse_args()
    return args.game.lower(), args.training

name_to_env_names = {
    "pong": "Pong-v4",
    "cartpole": "CartPole-v1"
}

requested_game, training = parse_args()
env_name = name_to_env_names[requested_game]

is_atari_game = requested_game != 'cartpole'

print(f"Selected Enviroment: {env_name}")

# Deep Q-Network approximates a state-value function in a Q-Learning framework with a neural network.
# It's similare to Q-Learning with some main differences:

# 1) Each step update is, potentially, used in many weights updates, which allow for greater data efficency.
# 2) Learning directly from consecutive samples is inefficient, due to the strong correlations between the samples; randoming the samples breaks these correlations and therefore
#   reduces the variance of the updates
# 3) When learning on-policy the current parameter determine the next data sample that the parameters are trained on. By using experience replay the behavior distribution is averaged 
#   over many of its previous states, smoothing out learning and avoid oscillations or diverrgence in the parameters.
# NOTE: when learning by experience replay. it is necessary to learn off-policy. (that's why of Q-Learning)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
def create_image_preprocessor():
    """
        IMAGE PREPROCESSING:
        Atari frame are 210x160 pixels with a 128 color palette, that a challenging amount of data to handle; we do few steps to reduce the dimensionality:
        1) We convert the frame into a gray-scale
        2) Down-sampling it to a 110x84 image. 
        3) Cropping an 84x84 region 
    """
    return transforms.Compose(transforms=[
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Resize((110, 84), antialias=True),
        transforms.CenterCrop((84,84))
    ])
    
image_transform = create_image_preprocessor()


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
initial_samples_size = 50000 # initial samples to collect with a random policy
linear_decay_episodes = 1000000
eps_greedy_start = 1
eps_greedy_end = 0.1
replay_buffer_size = 1000000
total_train_frames = 10000000
skip_frames = 4 if not env_name.startswith('SpaceInvaders') else 3
gamma = 0.99
batch_size = 32
obs_stack_dim = 4 if is_atari_game else 1

# Create atari game environment
if is_atari_game:
    env = gym.make(env_name, render_mode = None if training else 'human', frameskip=skip_frames)
else:
    env = gym.make(env_name, render_mode = None if training else 'human')


# Replica of the model presented in the paper. Is a standard CNN.
class DQNModel(torch.nn.Module):
    def __init__(self, obs_size : int, action_size : int) -> None:
        super().__init__()

        self.conv1_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=obs_size, out_channels=16, kernel_size=(8, 8), stride=4), # (N, 4, 84, 84) -> (N, 16, 20, 20)
            torch.nn.ReLU()
        ) 

        self.conv2_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=2), # (N, 4, 84, 84) -> (N, 16, 20, 20)
            torch.nn.ReLU()
        ) 

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=2592, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=action_size)
        )

    def forward(self, x):
        x = self.conv1_block(x)
        x = self.conv2_block(x)

        return self.classifier(x)
    

# eps scheduling.
def compute_eps(current_step, min, max, num_steps):
    alpha = current_step / num_steps
    alpha = np.clip(alpha, 0.0, 1.0)

    return min * (1 - alpha) + max * alpha


# init replay buffer
replay_buffer = ReplayBuffer(max_size=replay_buffer_size)

# init action-value Q
obs_dim = 1 # after image processing we will have 1 color channel.
action_dim = env.action_space.n

q_model = DQNModel(1, action_size=action_dim).to(device)
summary(q_model, input_size=(1,1,84,84))
