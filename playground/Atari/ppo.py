import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
import gymnasium as gym
from tqdm import tqdm
import tyro
from torchinfo import summary

@dataclass
class Args:
    # Name of this experiment, used to track the experiment on tensorboard  
    exp_name : str = "AgentZero"

    # Environment ID
    env_name : str = "BreakoutNoFrameskip-v4"

    # total timesteps of the experiment
    total_timesteps : int = 10000000

    # learning rate
    learning_rate : float = 2.5e-4

    # concurrent running agent to collect experiences
    concurrent_agents : int = 8

    # the number of steps to run in each environment before rollout
    rollout_num_steps : int = 128

    # the discount factor
    gamma : float = 0.99

    # number of mini batches
    minibatches_num : int = 4

    # the K epochs to update the policy
    update_epochs : int = 4

    # the surrogate clipping coeff
    clip_coef : float = 0.1

    # wether or not to clip vloss
    clip_vloss : bool = True

    # entropy coefficient
    beta : float = 0.1

    # value loss coefficient
    values_loss_coef : float = 0.5

    # the max norm of the gradient clipping
    max_grad_norm : float = 0.5

    ## atari params
    # frames to skip
    atari_skip_frames : int = 4

    # Max No-op at start episodes
    atari_noop_max : int = 30

    # Frames to stack together
    atari_frame_stack : int = 4


# envs setup
def make_env(env_id, noop : int = 30, skip : int = 4, frame_stack : int = 4):
    def thunk():
        # make the environment
        env = gym.make(env_id, frameskip=skip)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.AtariPreprocessing(env, noop_max=noop, frame_skip=skip, scale_obs=True)
        env = gym.wrappers.FrameStack(env, frame_stack)
        return env

    return thunk


# Replica of the model presented in the paper. Is a standard CNN.
class Model(torch.nn.Module):
    """
    This class defines a neural network model for reinforcement learning tasks.

    The model consists of a convolutional neural network followed by a fully connected layer.
    The convolutional layers extract features from the input observations, and the fully connected
    layer outputs the action probabilities and the value function.

    Args:
        obs_size (int): The size of the input observations.
        action_size (int): The number of possible actions.

    Attributes:
        net (torch.nn.Sequential): The convolutional neural network.
        actor (torch.nn.Linear): The fully connected layer that outputs the action probabilities.
        critic (torch.nn.Linear): The fully connected layer that outputs the value function.

    Methods:
        _layer_init(layer, std=np.sqrt(2), bias=0.0): Initializes the weights and biases of a layer using orthogonal initialization.
        get_critic(x): Computes the value function for the given input.
        forward(x, action=None): Computes the action probabilities and the value function for the given input.
    """

    def __init__(self, obs_size : int, action_size : int) -> None:
        super().__init__()

        self.net = torch.nn.Sequential(
            self._layer_init(torch.nn.Conv2d(in_channels=obs_size, out_channels=32, kernel_size=(8, 8), stride=4)), # (N, 4, 84, 84) -> (N, 32, 20, 20),
            torch.nn.ReLU(),
            self._layer_init(torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2)), # (N, 32, 20, 20) -> (N, 64, 9, 9)
            torch.nn.ReLU(),
            self._layer_init(torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1)), # (N, 64, 9 , 9) -> (N, 64, 7, 7)
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            self._layer_init(torch.nn.Linear(in_features=3136, out_features=512)), # 3136 == 64 * 7 * 7
            torch.nn.ReLU()
        )

        self.actor = self._layer_init(torch.nn.Linear(in_features=512, out_features=action_size), std=0.01)
        self.critic = self._layer_init(torch.nn.Linear(in_features=512, out_features=1), std=1)


    def _layer_init(self, layer, std : float =np.sqrt(2), bias : float = 0.0):
        """
        Initializes the weights and biases of a layer using orthogonal initialization.

        Args:
            layer (torch.nn.Module): The layer to be initialized.
            std (float): The standard deviation of the weight initialization.
            bias (float): The initial value of the biases.

        Returns:
            torch.nn.Module: The initialized layer.
        """

        # The 'std' parameter represents the gain value used in weight initialization to scale the weights appropriately.
        # more on https://pytorch.org/docs/stable/nn.init.html
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias)

        return layer

    def get_critic(self, x):
        """
        Computes the value function for the given input.

        Args:
            x (torch.Tensor): The input observations.

        Returns:
            torch.Tensor: The value function.
        """
        return self.critic(self.net(x))

    def forward(self, x, action = None):
        """
        Computes the action probabilities and the value function for the given input.

        Args:
            x (torch.Tensor): The input observations.
            action (torch.Tensor, optional): The actions to be used for computing the log probabilities.

        Returns:
            torch.Tensor: The selected actions.
            torch.Tensor: The log probabilities of the selected actions.
            torch.Tensor: The entropy of the action probabilities.
            torch.Tensor: The value function.
        """
        x = self.net(x)
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
def visualize_image(images):
        """
        Plots an array of gray images using matplotlib.pyplot.
        
        Args:
            images (numpy.ndarray): An array of gray images with shape [4][84][84].
        """
        # Check if the input is a valid numpy array
        if not isinstance(images, np.ndarray) or images.shape != (4, 84, 84):
            raise ValueError("Input must be a numpy array of shape [4][84][84].")
        
        # Create a figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(8, 8))
        
        # Plot each image in the subplots
        for i, ax in enumerate(axes.flat):
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')
        
        # Adjust the spacing between subplots
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        
        # Display the plot
        plt.show()
    
if __name__ == "__main__":
    # parse cmdline and get arguments for the training
    args = tyro.cli(Args)

    # create a vector of environment that we will use to sample data.
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_name, args.atari_noop_max, args.atari_skip_frames, args.atari_frame_stack) for i in range(args.concurrent_agents)]
    )

    # compute some variables that will be use for training
    batch_size = args.concurrent_agents * args.rollout_num_steps

    assert batch_size % args.minibatches_num == 0
    mini_batch_size = batch_size // args.minibatches_num

    # total iterations num
    iterations_num = args.total_timesteps // batch_size

    # create tensorboard writer to plot training data
    writer = SummaryWriter(f"runs/{args.exp_name}" if args.exp_name is not None else None)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # create our initial model
    model = Model(envs.single_observation_space.shape[0], envs.single_action_space.n).to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-5)


    # init training variables
    global_step = 0
    start_time = time.time()

    # initial observations
    obs, _ = envs.reset()
    obs = torch.tensor(obs, device=device)

    # print the model
    summary(model, input_size=obs.shape)
    
    #visualize_image(obs[0].cpu().numpy())
    for iterations in tqdm(range(iterations_num), desc='Training Progress'):

        observations = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        for step in range(args.rollout_num_steps):
            global_step += args.concurrent_agents

            with torch.inference_mode():
                action, log_prob, _, critic = model(obs)
                values.append(critic)

            actions.append(action)
            log_probs.append(log_prob)

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            rewards.append(reward)

            next_obs = torch.tensor(next_obs, device=device)
            done = np.logical_or(terminations, truncations)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        #print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("graphs/episodes_rewards", info["episode"]["r"], global_step)
                        writer.add_scalar("graphs/episodes_length", info["episode"]["l"], global_step)

        

    