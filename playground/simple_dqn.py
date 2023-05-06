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
from itertools import count

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

# utility function to track scores
def running_mean(x, samples=50):
    kernel = np.ones(samples)
    conv_len = x.shape[0] - samples
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i : i + samples]
        y[i] /= samples
    return y

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
batch_size = 128
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

steps_done = 0

@torch.inference_mode()
def select_action(model : torch.nn.Module, obs, eps:float):    
    if np.random.random() > eps:
        return model(obs).argmax()
    else:
        return torch.tensor(env.action_space.sample(), device=device)
    

obs = torch.from_numpy(obs).to(device)

def optimize(model : torch.nn.Module, replay_buffer : ReplayBuffer):
    """
        For our training update rule, we'll use the fact that every Q function for some policy obeys to the Bellman Equation:
                    Q(s_t, a) = r + yQ(s_t+1, pi(s_t+1))
        The difference between the left and right sides of the equality is called temporal differnece error:
                    t_d_e = Q(s, a) - (r + y*max_a(Q(s_t+1, a))) 
    """
    if(len(replay_buffer) < batch_size):
        return
    
    batch = replay_buffer.sample(batch_size=batch_size)

    # compute if state are final or not.
    # ...

    # Compute Q(s_t, a) - the model compute Q(s_t) for each action. 
    # then we select the columns of actions taken. These are the actions taken from the policy_model

    # ...

    # Compute V(s_t+1) for all next states.
    # Expected values of final next states is zero.
    # Expected values of non final next states are computed based on stale-policy.

    # ...

    # Compute expected Q values

    # ...

    # Compute loss

    # ...

    # ZeroGrad. Optimize

    # ...

    # Clip grad. Step.

    # ...

# fill replay buffer first.
def initial_fill_replay_buffer(replay_buffer : ReplayBuffer):
    progress_bar = tqdm(total = batch_size * 2, desc='Initialize replay buffer with random experiences.', position=0)
    progress_bar.set_description_str('Filling replay_buffer with random experiences')

    initial_replay_buffer_size = len(replay_buffer)
    for t in count():
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        
        for _ in count():
            action = select_action(policy_model, state, 1.0) # always random action
            obs , rew, terminated, truncated, info = env.step(action.item())

            rew = torch.tensor(rew, device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(obs, dtype=torch.float32, device = device)

            replay_buffer.add((state, action, rew, next_state))

            state = next_state

            initial_replay_buffer_size = len(replay_buffer)

            progress_bar.update(1)

            if done or initial_replay_buffer_size >= batch_size * 2:
                break

        if  initial_replay_buffer_size >= batch_size * 2:
            break

initial_fill_replay_buffer(replay_buffer=replay_buffer)

num_episodes = 600
steps_done = 0
align_models_every_nstep = 4
scores = []
outer = tqdm(total=num_episodes, desc = 'Episodes', position=0)

for i in tqdm(range(num_episodes)):
    state, info = env.reset()
    state = torch.tensor(state, device=device)

    for t in count():
        # compute eps for current epoch
        eps = compute_eps(steps_done, eps_start, eps_end, eps_decay)
        steps_done += 1

        # select action and step the environment receive reward and next state.
        action = select_action(policy_model, state, eps)
        obs, rew, terminated, truncated, info = env.step(action.item())
        rew = torch.tensor(rew, device=device)
        done = terminated or truncated

        next_state = torch.tensor(obs, device=device, dtype=torch.float32) if not done else None 

        # add current experience to the replay buffer.
        replay_buffer.add((state, action, rew, next_state))

        state = next_state

        # One step optimization of the model. At this point we already have at least batch_size*2 elements in the replay buffer.
        optimize(policy_model, replay_buffer=replay_buffer)

        # Check if in this epoch we need to align the weights.
        if steps_done % align_models_every_nstep == 0:
            # align the model weights
            target_model.load_state_dict(policy_model.state_dict())
             
        if done:
            scores.append(t+1)
            outer.set_description_str(f"Avg score: {running_mean(scores)}")
            outer.update(1)
            break


