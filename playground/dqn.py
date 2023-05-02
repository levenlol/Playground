import torch
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from torchinfo import summary
from tqdm import tqdm

import argparse

# Parse commandline for game name and other infos.
def parse_args():
    parser = argparse.ArgumentParser(description='Parse command line arguments.')
    parser.add_argument('--game', type=str, default='Pong', help='Name of the game to play')
    parser.add_argument('--training', action='store_true', help='Enable training mode')
    args = parser.parse_args()
    return args.game.lower(), args.training

name_to_env_names = {
    "pong": "Pong-v4"
}

requested_game, training = parse_args()
env_name = name_to_env_names[requested_game]

print(f"Selected Enviroment: {env_name}")

# Create atari game environment
env = gym.make(env_name, render_mode = None if training else 'human')
a =5

# Deep Q-Network approximates a state-value function in a Q-Learning framework with a neural network.
# It's similare to Q-Learning with some main differences:

# 1) Each step update is, potentially, used in many weights updates, which allow for greater data efficency.
# 2) Learning directly from consecutive samples is inefficient, due to the strong correlations between the samples; randoming the samples breaks these correlations and therefore
#   reduces the variance of the updates
# 3) When learning on-policy the current parameter determine the next data sample that the parameters are trained on. By using experience replay the behavior distribution is averaged 
#   over many of its previous states, smoothing out learning and avoid oscillations or diverrgence in the parameters.
#   NOTE: when learning by experience replay. it is necessary to learn off-policy. (that's why of Q-Learning)

# IMAGE PREPROCESSING:
# Atari frame are 210x160 pixels with a 128 color palette, that a challenging amount of data to handle; we do few steps to reduce the dimensionality:
# 1) We convert the frame into a gray-scale
# 2) Down-sampling it to a 110x84 image. 
# 3) Cropping an 84x84 region 

