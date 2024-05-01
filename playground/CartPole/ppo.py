import torch
import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt

from dataclasses import dataclass

@dataclass
class Args:
    # Name of the environment
    env_name : str = 'CartPole-v1'

    # Total episodes of the experiment
    total_epochs : int = 5000

    # learning rate
    learning_rate : float = 2e-5

    # number of concurrent agents
    concurrent_agents : int = 4

    # number of steps for policy rollout
    num_steps : int = 128

    # discount factor
    gamma : float = 0.99

    # mini batches num
    num_minibatches : int = 4

    # the surrogate clipping coefficient
    clip_coef : float = 0.2

    # the K epochs to update the policy
    update_epochs : int = 4

    # entropy coefficient
    beta : float = 0.01

    # value function coefficient
    vf_coef : float = 0.5

    # maximum value for gradient clipping
    max_grad_norm : float = 0.5


num_iteration = 0

# todo: override from cli
args = Args()

batch_size = int(args.concurrent_agents * args.num_steps)
minibatch_size = int(batch_size // args.num_minibatches)

class MLP(torch.nn.Module):
    def __init__(self, obs_size : int, action_size : int, neurons : int = 64) -> None:
        super().__init__()

        self.critic = torch.nn.Sequential(
            torch.nn.Linear(obs_size, neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(neurons, neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(neurons, 1)
        )

        self.actor = torch.nn.Sequential(
            torch.nn.Linear(obs_size, neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(neurons, neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(neurons, action_size)
        )

    def forward(self, x, action = None):
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    

device = 'cuda' if torch.cuda.is_available else 'cpu'


def compute_returns(rewards : np.array, dones : np.array, gamma = 0.99):
    """
    Computes the discounted return for the given sequence of rewards.
    :param rewards: The sequence of rewards.
    :param dones: The sequence of done flags (True if episode terminated, False otherwise).
    :param gamma: The discount factor .
    :returns: The discounted return for each time step.
    """

    T = len(rewards)
    returns = np.zeros_like(rewards, dtype=float)

    R = 0
    for t in reversed(range(T)):
        R = rewards[t] + gamma * R * (1 - np.array(dones[t], dtype=np.int32))
        returns[t] = R

    return returns


# envs setup
def make_env(env_id, ):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

envs = gym.vector.SyncVectorEnv(
    [make_env(args.env_name) for i in range(args.concurrent_agents)]
)

assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action-space is supported"

obs_size = envs.single_observation_space.shape[0]
action_space = envs.single_action_space.n

model = MLP(obs_size, action_space, neurons=64).to(device)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-5)


total_steps = 0

obs, _ = envs.reset()
obs = torch.tensor(obs).to(device)
done = np.full(args.concurrent_agents, False)

# training loop
counters = np.zeros(args.concurrent_agents)
scores = []


for epoch in range(args.total_epochs):
    # prepare buffer to store data. maybe is worth pre allocate, and convert to tensors.
    observations = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []

    # collect episode data
    for step in range(args.num_steps):
        total_steps += args.concurrent_agents

        # store obs and dones
        observations.append(obs)
        dones.append(done)

        with torch.inference_mode():
            # run the model and get action, logprob and critic
            action, log_prob, _, value = model(obs)
            value = value.flatten()

        # store actions and values
        actions.append(action)
        values.append(value)

        # store log probs
        log_probs.append(log_prob)

        # step the envs
        obs, reward, terminations, truncations, _ = envs.step(action.cpu().numpy())
        obs = torch.tensor(obs).to(device)

        done = np.logical_or(terminations, truncations)

        rewards.append(reward)

        # record some statics
        counters += 1 - done
        for i in range(args.concurrent_agents):
            if done[i]:
                scores.append(counters[i])
                counters[i] = 0.0

    if len(scores) > 0:
        mean = np.mean(scores[np.max([-len(scores), -50]):-1]) if len(scores) > 1 else scores[0]
        print(f"Epoch: {epoch+1} Reward {mean:.2f}")






    # Perform training
    
    # preparaing tensors, flattens data
    observations = torch.stack(observations).reshape((-1,) + envs.single_observation_space.shape)
    log_probs = torch.stack(log_probs).reshape(-1)
    actions = torch.stack(actions).reshape(-1)
    returns = torch.from_numpy(compute_returns(rewards, dones)).to(torch.float32).to(device).reshape(-1)
    values = torch.stack(values).reshape(-1)
    advantages = returns - values

    # randomize 
    indices = np.arange(batch_size)

    for update_epoch in range(args.update_epochs):
        np.random.shuffle(indices)

        # compute mini_batch start/end
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size

            mini_indices = indices[start:end]

            # run the optimized model to get new data. action isnt needed here
            _, new_log_prob, entropy, new_critic = model(observations[mini_indices], actions[mini_indices])
            log_ratio = new_log_prob - log_probs[mini_indices]
            ratio = log_ratio.exp()

            with torch.inference_mode():
                # Approximate KL based on: http://joschu.net/blog/kl-approx.html
                #old_kl = (-log_ratio).mean()
                kl = ((ratio - 1.0) - log_ratio).mean()
                #clip [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            # normalize advantages
            mb_advantages = advantages[mini_indices]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            policy_loss1 = -mb_advantages * ratio
            policy_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            policy_loss = torch.max(policy_loss1, policy_loss2).mean()

            # Value loss
            value_loss_unclipped = (new_critic.squeeze() - returns[mini_indices]) ** 2
            value_clipped = values[mini_indices] + torch.clamp(new_critic - values[mini_indices], -args.clip_coef, args.clip_coef)
            value_loss_clipped = (value_clipped - returns[mini_indices]) ** 2

            value_loss_max = torch.max(value_loss_unclipped, value_clipped)
            value_loss = 0.5 * value_loss_max.mean()

            # Entropy loss
            entropy_loss = entropy.mean()
            loss = policy_loss + value_loss * args.vf_coef - args.beta * entropy_loss

            # optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # clip grads

            optimizer.step()

        















