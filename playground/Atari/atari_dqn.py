# IMPLEMENTATION OF: Playing Atari with Deep Reinforcement Learning
# reference: https://arxiv.org/pdf/1312.5602.pdf and https://arxiv.org/pdf/1711.07478.pdf and https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

# Deep Q-Learning uses a deep neural network to approximate the different Q-values for each possible action at a state

# In Deep Q-Learning we create a loss function that compares our Q-value prediction and the Q-target to update model weights.

# y = r + gamma * maxQ(a, s, teta-) # Q-TARGET
# y - Q(a, s, teta)  # Q-LOSS

# We are gonna use a different methods to stabilie the training process:
#
# 1) Experience Replay
# 2) Fixed Q-target to stabilize the training
# 3) Double Deep Q-Learning to handle the problem of the overestimation of Q-Values

# In depth:

# EXPERIENCE REPLAY
# Experience replay have 2 main purposes:

# 1) Make more efficient use of the experiences during the training. 
# Usually, in online reinforcement learning, the agent interacts with the environment,
# gets experiences (state, action, reward, and next state), learns from them (updates the neural network), and discards them. 
# This is not efficient.
# Experience Replay helps by using the experiences of the training more efficiently. We use a replay buffer that saves experience samples
# that we can reuse during the training => allow the agent to learn from the same experiences multiple times.

# 2) Avoid forgetting previous experience (catastrophic forgetting).

# FIXED Q-TARGET

# When we compute the TD error (loss) we obtain it subtracting the difference between the TD-target (Qtarget) and the current Q-Value (estimation of Q)
# We don't have any idea of the real TD Target. we need to estimate it.
# Using the Bellman equation, we saw that the TD target is just the reward of taking that action at that state plus the discounted highest Q value for the next state.

# The problem is that we are using the same weights to estimate the TD target AND the Q-Value. There is significant correlation between the TD and the parameters we are changing
# It's like we are moving the target when we are reaching it -> oscillation in training.

# To solve
# 1) Use a separate network with fixed parameters for estimating TD target
# 2) Copy the parameters from our Deep Q-network every C steps to update the target network.

# DOUBLE DQN
# This method handles the problem of overestimating Q-values

# recall: New_V_state = Old_V_State + learning_rate * [Reward + gamma * V(St+1) - V(St)]

# How are we sure that the best action is the action with the highest Q-Value ?
# The accuracy of Q-values depends on waht action we tried and WHAT neighboring states we explored

# We dont have such infos at the beginning of the training, if non-optimal actions gives a higher Q value than the optimal best action,
# the learning will be complicated.

# Solution:
# 1) Use our DQN Network to select the best action to take for the next state
# 2) Use our Target Network to calculate the target Q-value of taking that action at the next state


# TODO: consider implementing Prioritized Experience Replay and Dueling Deep Q-Learning.

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from dataclasses import dataclass
import gymnasium as gym
from torchvision import transforms
from tqdm import tqdm
from itertools import count

# Replay buffer simples implementations.
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
    
    
@dataclass
class Args:
    # environment_id
    env_id : str = 'BreakoutNoFrameskip-v4'
    # number of frames to skip
    env_skip_frames : int = 4
    # number of frames to stack
    env_stack_frames : int = 4

    # tensor board experiment name
    tensorboard_name : str = "AgentZero"

    # total timesteps
    total_timesteps : int = 10000000
    # learning rate
    learning_rate : float = 1e-4
    
    # replay buffer memory size
    buffer_size : int = 1000000

    # discount factor
    gamma : float = 0.99

    # target network update rate
    tau : float = 1.0

    # target network update frequency
    target_net_update_frequency : int = 1000

    # batch size
    batch_size = 32

    # epsilon value at start of training
    epsilon_start : float = 1.0

    # epsilon value at end of exploration
    epsilon_end : float = 0.1

    # steps to go from epsilon start to end
    epsilon_timesteps : int = 1000000

    # number of data collected before train starts
    preliminary_data_num : int = 80000

    # frequency of train
    train_frequency : int = 4


# TODO: override from cli
args = Args()

# Replica of the model presented in the paper. Is a standard CNN.
class DQNModel(torch.nn.Module):
    def __init__(self, obs_size : int, action_size : int) -> None:
        super().__init__()

        self.conv1_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=obs_size, out_channels=32, kernel_size=(8, 8), stride=4), # (N, 4, 84, 84) -> (N, 32, 20, 20),
            torch.nn.ReLU()
        )

        self.conv2_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2), # (N, 32, 20, 20) -> (N, 64, 9, 9)
            torch.nn.ReLU()
        )

        self.conv3_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1), # (N, 64, 9 , 9) -> (N, 64, 7, 7)
            torch.nn.ReLU()
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=3136, out_features=512), # 3136 == 64 * 7 * 7
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=action_size)
        )

    def forward(self, x):
        x = self.conv1_block(x)
        x = self.conv2_block(x)
        x = self.conv3_block(x)

        return self.classifier(x)
    

@torch.inference_mode()
def select_action(model: torch.nn.Module, obs, eps_greedy : float):
    """
    Function to select an action based on the given model and observation using an epsilon-greedy strategy.

    Parameters:
    - model (torch.nn.Module): The neural network model used for action selection.
    - obs: The observation input to the model for action selection.
    - eps_greedy (float): The epsilon value for the epsilon-greedy strategy.

    Returns:
    - action: The selected action based on the epsilon-greedy strategy.

    Note:
    - This function is used during inference to select actions based on the model's predictions.
    - The epsilon-greedy strategy balances exploration and exploitation in reinforcement learning.
    """
    if np.random.random() > eps_greedy:
        return model(obs.unsqueeze(0)).argmax()
    else:
        return torch.tensor(env.action_space.sample())


def preprocess_image(image):
    """
    Preprocess the Atari game image to convert it into a tensor of size (4, 84, 84).

    Parameters:
    - image: The input image of size (4, 210, 160, 3).

    Returns:
    - tensor: The preprocessed tensor of size (4, 84, 84).
    """
    image_array = np.array(image)
    t = transforms.Compose(transforms=[
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.CenterCrop((175,150)),
        transforms.Resize((84, 84)),
        transforms.ToTensor()
    ])

    tensor = torch.zeros([4, 84, 84])
    for i in range(image_array.shape[0]):
        tensor[i] = t(image_array[i])

    return tensor

def visualize_frame(image):
    """
    Display a single frame from an image array.

    Parameters:
    image (numpy.ndarray): Array containing frames/images.
    """
    plt.imshow(image, cmap='gray')
    plt.show()


# make the environment
env = gym.make(args.env_id, frameskip=args.env_skip_frames)
env = gym.wrappers.FrameStack(env, args.env_stack_frames)

def compute_eps(current_step, min, max, num_steps):
    """
    Compute the epsilon value for a given step in a learning process.

    Parameters:
    current_step (int): The current step in the learning process.
    min (float): The minimum value for the epsilon.
    max (float): The maximum value for the epsilon.
    num_steps (int): The total number of steps in the learning process.

    Returns:
    float: The calculated epsilon value for the current step.
    """
    alpha = current_step / num_steps
    alpha = np.clip(alpha, 0.0, 1.0)
    return min * (1 - alpha) + max * alpha


def initial_fill_replay_buffer(replay_buffer : ReplayBuffer, env : gym.Env, size : int):
    """
    Fill the replay buffer with initial experiences.

    This function is used to populate the replay buffer with a set of initial experiences before
    the training process begins. This helps the agent learn from a diverse set of experiences
    from the start, rather than relying solely on the initial interactions with the environment.

    Parameters:
    replay_buffer (ReplayBuffer): The replay buffer to be filled.
    env (gym.Env): the environment to sample experience from.
    size (int): The desired size of the initial fill for the replay buffer.
    """

    progress_bar = tqdm(total=size, desc='Initialize replay buffer with random experiences.', position=0)
    progress_bar.set_description_str("Filling replay buffer with random experiences")

    while len(replay_buffer) < size:
        obs, info = env.reset()
        
        for _ in count():
            obs = preprocess_image(obs)

            action = select_action(None, obs, 1.0) #always random action
            next_obs, rew, terminated, truncated, info = env.step(action.item())

            done = terminated or truncated
            next_obs = preprocess_image(next_obs) if not done else None 
            

            replay_buffer.add((obs, action, rew, next_obs, done))

            obs = next_obs
            progress_bar.update(1)

            if done or len(replay_buffer) >= size:
                break

    


if __name__ == "__main__":
    # we will use tensorboard to track our experiment learning process
    writer = SummaryWriter(log_dir=f"runs/{args.tensorboard_name}" if args.tensorboard_name is not None else None)

    # log hyperparameters
    writer.add_text("Hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # init target net and q network
    q_network = DQNModel(args.env_stack_frames, env.action_space.n).to(device)
    target_network = DQNModel(args.env_stack_frames, env.action_space.n).to(device)

    # align models weights to start
    target_network.load_state_dict(q_network.state_dict())

    # define optimizer
    optimizer = torch.optim.AdamW(q_network.parameters(), lr=args.learning_rate)

    # create replay buffer and init it with random initial experiences
    replay_buffer = ReplayBuffer(args.buffer_size)
    #initial_fill_replay_buffer(replay_buffer, env, args.preliminary_data_num)

    print("done")