import torch
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from torchinfo import summary
from tqdm import tqdm
import torchvision
from torchvision import transforms
from itertools import count
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import random

import argparse

# Utils
import logging
def disable_logging(func):
    def wrapper(*args, **kwargs):
        # Disable logging
        logging.disable(logging.CRITICAL)

        # Call the original function
        result = func(*args, **kwargs)

        # Re-enable logging
        logging.disable(logging.NOTSET)

        return result
    return wrapper

# IMPLEMENTATION OF: Playing Atari with Deep Reinforcement Learning
# https://arxiv.org/pdf/1312.5602.pdf and https://arxiv.org/pdf/1711.07478.pdf and https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf


# Parse commandline for game name and other infos.
def parse_args():
    parser = argparse.ArgumentParser(description="Deep Q-Network (DQN) algorithm with PyTorch and Atari Games (Gym Env).")
    parser.add_argument("--game", type=str, default="Pong", help="Name of the game to play")
    parser.add_argument("--training", action="store_true", default=True, help="Enable training mode")
    parser.add_argument("--tau", type=float, default=1, help="Value of tau for soft target network update (default: 1)") # 1 for hard update, <1 for soft updates.
    parser.add_argument("--align-models-every", type=int, default=1, help="Number of steps between aligning the policy and target networks (default: 10000)")
    args = parser.parse_args()
    return args.game.lower(), args.training, args.tau, args.align_models_every

name_to_env_names = {"pong": "Pong-v4"}

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

requested_game, training, TAU, align_models_every_nstep = parse_args()
env_name = name_to_env_names[requested_game]

print(f"Selected Enviroment: {env_name}")

initial_samples_size = 50000  # initial samples to collect with a random policy
linear_decay_episodes = 1000000
eps_greedy_start = 1
eps_greedy_end = 0.1
replay_buffer_size = 1000000
total_train_frames = 10000000
skip_frames = 4 if not env_name.startswith("SpaceInvaders") else 3
gamma = 0.99
batch_size = 32
obs_stack_dim = 4
learning_rate = 0.00025
gradient_momentum = 0.95
squared_gradient_momentum = 0.95
min_squared_gradient = 0.1


# Deep Q-Network approximates a state-value function in a Q-Learning framework with a neural network.
# It's similare to Q-Learning with some main differences:

# 1) Each step update is, potentially, used in many weights updates, which allow for greater data efficency.
# 2) Learning directly from consecutive samples is inefficient, due to the strong correlations between the samples; randoming the samples breaks these correlations and therefore
#   reduces the variance of the updates
# 3) When learning on-policy the current parameter determine the next data sample that the parameters are trained on. By using experience replay the behavior distribution is averaged
#   over many of its previous states, smoothing out learning and avoid oscillations or diverrgence in the parameters.
# NOTE: when learning by experience replay. it is necessary to learn off-policy. (that's why of Q-Learning)

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_image_preprocessor():
    """
    IMAGE PREPROCESSING:
    Atari frame are 210x160 pixels with a 128 color palette, that a challenging amount of data to handle; we do few steps to reduce the dimensionality:
    1) We convert the frame into a gray-scale
    2) Down-sampling it to a 110x84 image.
    3) Cropping an 84x84 region
    """
    return transforms.Compose(
        transforms=[
            transforms.ToPILImage(),
            transforms.Resize((110, 84), antialias=True),
            transforms.CenterCrop((84, 84)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ]
    )


image_transform = create_image_preprocessor()


# Replay Buffers are a central part of off-policy RL algorithms.

# A replay buffer is a data structure that stores experiences of the agent as it interacts with the environment.
# Each experience is typically represented as a tuple containing the current state, the action taken, the resulting reward, and the next state.
# The replay buffer can be thought of as a dataset of experiences that the agent can use to learn from.
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque([], maxlen=max_size)

    # Add a new experience to the buffer.
    def add(self, experience):
        self.buffer.append(experience)

    # Samples a batch of experiences from the buffer of size batch_size and returns the experiences.
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    # This method returns the number of experiences currently stored in the buffer.
    def __len__(self):
        return len(self.buffer)


# Simple utils rollout buffer. This is used to stack N-frames together.
class RolloutBuffer():
    def __init__(self, max_size, device="cpu"):
        self.buffer = deque([], maxlen=max_size)
        self.device=device

    # init the buffer to various copies of same experience.
    def fill(self, experience):
        self.buffer.clear()
        for i in range(self.buffer.maxlen):
            self.add(experience)

    # Add a new experience to the buffer.
    def add(self, experience):
        self.buffer.append(experience.to(self.device))

    # get buffers. stacked together
    def get(self):
        return torch.cat(tuple(trajectory_container.buffer))

# Create atari game environment
env = gym.make(
    env_name, render_mode=None if training else "human", frameskip=skip_frames
)


# Replica of the model presented in the paper. Is a standard CNN.
class DQNModel(torch.nn.Module):
    def __init__(self, obs_size: int, action_size: int) -> None:
        super().__init__()

        self.conv1_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=obs_size, out_channels=32, kernel_size=(8, 8), stride=4
            ),  # (N, 4, 84, 84) -> (N, 32, 20, 20)
            torch.nn.ReLU(),
        )

        self.conv2_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2
            ),  # (N, 32, 20, 20) -> (N, 64, 9, 9)
            torch.nn.ReLU(),
        )

        self.conv3_block = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1
            ),  # (N, 64, 9, 9) -> (N, 64, 7, 7)
            torch.nn.ReLU(),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=3136, out_features=512),  # 3136 = 64*7*7
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=action_size),
        )

    def forward(self, x):
        x = self.conv1_block(x)
        x = self.conv2_block(x)
        x = self.conv3_block(x)

        return self.classifier(x)


# select action with eps_greedy param.
@torch.inference_mode()
def select_action(model: torch.nn.Module, obs, eps_greedy: float):
    if np.random.random() > eps_greedy:
        return model(obs.unsqueeze(0).to(device)).argmax().to('cpu')
    else:
        return torch.tensor(env.action_space.sample())


# eps scheduling.
def compute_eps(current_step, min, max, num_steps):
    alpha = current_step / num_steps
    alpha = np.clip(alpha, 0.0, 1.0)
    return min * (1 - alpha) + max * alpha


# init action-value Q
obs_dim = (4, 84, 84)  # after image processing we will have 1 color channel.
action_dim = env.action_space.n

# create models.
policy_model = DQNModel(obs_dim[0], action_size=action_dim).to(device)
target_model = DQNModel(obs_dim[0], action_size=action_dim).to(device)

target_model.load_state_dict(policy_model.state_dict())

summary(policy_model, input_size=(batch_size, *obs_dim))

def sample_mean(arr, num_samples):
    sample_size = len(arr) // num_samples
    mean_arr = np.zeros(num_samples)
    for i in range(num_samples):
        start = i * sample_size
        end = start + sample_size
        mean_arr[i] = np.mean(arr[start:end])
    return mean_arr

# optimizer = torch.optim.RMSprop(policy_model.parameters(), lr=learning_rate, momentum=gradient_momentum, alpha=squared_gradient_momentum, eps=0.01)
optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
steps_done = 0

# init replay buffer
replay_buffer = ReplayBuffer(replay_buffer_size)
trajectory_container = RolloutBuffer(obs_dim[0], "cpu")

# fill replay buffer first.
def initial_fill_replay_buffer(replay_buffer: ReplayBuffer, size: int):
    progress_bar = tqdm(
        total=size, 
        desc="Initialize replay buffer with random experiences.", 
        position=0
    )
    progress_bar.set_description_str("Filling replay_buffer with random experiences")
    progress_bar.update(len(replay_buffer))

    while len(replay_buffer) < size:
        state, info = env.reset()
        state = image_transform(state)
        trajectory_container.fill(state)

        for _ in count():
            current_state = trajectory_container.get()

            action = select_action(None, current_state, 1.0)  # always random action
            obs, rew, terminated, truncated, info = env.step(action.item())

            rew = torch.clamp(torch.tensor(rew), -1, 1)
            done = terminated or truncated

            if done:
                next_state = None
            else:
                trajectory_container.add(image_transform(obs))
                next_state = trajectory_container.get() 

            replay_buffer.add((current_state, action, rew, next_state))
            progress_bar.update(1)

            if done or len(replay_buffer) >= size:
                break

def optimize(model: torch.nn.Module, stale_model: torch.nn.Module, replay_buffer: ReplayBuffer, device):
    """
        For our training update rule, we'll use the fact that every Q function for some policy obeys to the Bellman Equation:
                    Q(s_t, a) = r + yQ(s_t+1, pi(s_t+1))
        The difference between the left and right sides of the equality is called temporal differnece error:
                    t_d_e = Q(s, a) - (r + y*max_a(Q(s_t+1, a))) 
    """
    # If we dont have enough experiences in the replay buffer just skip optimization
    if(len(replay_buffer) < batch_size):
        return
    
    # Get a batch to perform model optimization.
    batch = replay_buffer.sample(batch_size)
    batch = Transition(*zip(*batch))

    states = torch.stack(batch.state).to(device)
    actions = torch.stack(batch.action).unsqueeze(1).to(device) # unsqueeze to match dimension for gather later.
    rewards = torch.stack(batch.reward).to(device)
    
    # Next states requires special handling because it might be None. 
    # In order to accomplish that we need a mask that tells us if the tensor at given index in the batch is None
    next_states = torch.stack([t for t in batch.next_state if t is not None]).to(device)
    next_states_mask = torch.tensor([t is not None for t in batch.next_state]).to(device)

    # Compute Q(s_t, a) - the model compute Q(s_t) for each action. 
    # then we select the columns of actions taken. These are the actions taken from the policy_model
    # Store the Q(s_t) for the selected action. might or not might be the "optimal" policy, depends on the chosen action (and eps-greedy)
    q_values = model(states).gather(1, actions)

    # Compute V(s_t+1) for all next states.
    # Recall V(s_t) = maxa(Q(s_t, a))
    # Expected values of final next states is zero.
    # Expected values of non final next states are computed based on stale-policy.
    next_state_values = torch.zeros(batch_size, device=device)

    # we need to compute the grads wrt policy weights, not the stale one.
    with torch.inference_mode():
        v_values = stale_model(next_states).amax(dim=1) # 
        next_state_values[next_states_mask] = v_values # fill according to the mask. 0 otherwise

    # Compute expected Q values
    expected_q_values = rewards + (next_state_values * gamma)

    # Compute Loss
    # TODO: clamp the error
    criterion = torch.nn.MSELoss()
    loss = criterion(q_values, expected_q_values.unsqueeze(1))
    
    # ZeroGrad. backward. Clip. Optimize
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()


@disable_logging
def align_models(dict):
    #target_model.load_state_dict(policy_model.state_dict())
    # Soft update of the target network's weights. if TAU is 1 it is the same as doing an Hard-update
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_model.state_dict()
    for key in dict:
        target_net_state_dict[key] = dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target_model.load_state_dict(target_net_state_dict)

initial_fill_replay_buffer(replay_buffer, initial_samples_size)

progress_bar = tqdm(total=total_train_frames, desc="Training progress.", position=0)
scores = []

steps_done = 0
while steps_done < total_train_frames:
        # Reset the environment to initial state
        obs, info = env.reset()
        # Perform image pre-processing
        obs = image_transform(obs)
        # Init stacked frame to a copy of the initial frame.
        trajectory_container.fill(obs)
        score = 0

        for _ in count():
            # get current state, it is composed of the last 4 'played' frame
            current_state = trajectory_container.get()
            
            # Compute eps-greedy probability and select an action
            eps = compute_eps(steps_done, eps_greedy_start, eps_greedy_end, linear_decay_episodes)
            action = select_action(policy_model, current_state, eps)
            # Interact with the environment and get new state and reward
            obs, rew, terminated, truncated, info = env.step(action.item())
            score += rew
            rew = torch.clamp(torch.tensor(rew), -1, 1)

            done = terminated or truncated

            if done:
                next_state = None
            else:
                trajectory_container.add(image_transform(obs))
                next_state = trajectory_container.get()

            # Add new state-transition to the replay buffer.
            replay_buffer.add((current_state, action, rew, next_state))
            
            # Perform q-model (policy) optimization
            optimize(policy_model, target_model, replay_buffer, device)

            # Align the two network weights. Either soft update or hard update.
            steps_done += 1

            if (steps_done % align_models_every_nstep == 0):
                align_models(policy_model.state_dict())
            
            # Update training informations.
            progress_bar.update(1)
            if done:
                scores.append(score)
                scores_num = len(scores)
                avg_score = np.mean(scores[np.max([-scores_num, -50]):-1]) if scores_num > 1 else score
                progress_bar.set_description_str(f"Avg Score: {avg_score:.2f}")
                break


