import torch
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from torchinfo import summary
from tqdm import tqdm


# Vanilla Policy Gradient (aka Reinforce Algorithm) is the first algorithm of the Reinforcement Learning Family.
# It is quite simple and we will try to use to solve the simple cart-pole gym environment.

# Action space is [0, 1] (left-right)
# Observation space [Cart Position(-4.8, 4.8), Cart Velocity(-inf, inf), Pole Angle(-0.418, 0.418), Pole Angular Velocity(-inf, inf)]
# Rewards: +1 every step

env = gym.make("CartPole-v1")
obs_size = env.observation_space.shape[0]
action_space = env.action_space.n

# The first thing we need to define is the Trajectory.
# Trajectory is a sequence of (state-action-rewards). The length of the trajectory is called Horizon H.

# T = (s0, a0, r1, s1, a1, r2, ..., aH, rH+1)
# Note: we will ignore the rewards for the first implementation and for this particular problem.

# The second thing we need is the Return for a Trajectory. It is calculated as the sum reward from the trajectory t:
# R(t) = (G0, G1, ..., GH)

# The total return at time step k for the transition k (sk, ak, rk+1)
# Gk <- sum(gamma^(t-k-1) * Rk)
# It is the return we expect to collect from timestep k until the end of the trajectory. (discounted)

# The GOAL of the algorithm is to maximize the expected return U(teta)
# U(teta) = Sum(P(t;teta)R(t))

# Gradient ascent
# Teta <- teta + alpha * U(teta)'

# Create an MLP model

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cpu"


class MLP(torch.nn.Module):
    def __init__(self, obs_size, action_size, hidden_dim=256) -> None:
        super().__init__()

        self.obs_size = obs_size
        self.action_size = action_size
        self.hidden_dim = 256

        # head of the model
        self.first_linear = torch.nn.Linear(obs_size, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.second_linear = torch.nn.Linear(hidden_dim, action_size)

    def forward(self, x):
        x = self.first_linear(x)
        x = self.relu(x)
        x = self.second_linear(x)
        return x


model = MLP(obs_size, action_space).to(device)
summary(model, input_size=([1, obs_size]))

lr = 0.002
gamma = 0.99

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

score = []


def train(
    model: torch.nn.Module,
    env: gym.Env,
    optimizer: torch.optim.Optimizer,
    device=device,
    horizon=500,
    episodes=500,
):
    """
    Perform the training. Algorithm Reinforce:
    for each episode {s1, a1, r2, ..., st-1, at-1, rt} ~ p0 do
        for t = 1 to T-1 do:
            teta <- teta + alpha*logp0(st, at) vt

    Recall the Policy Gradient Theorem:
    J0'= Ep0[logp0(s, a) * Qp0(s, a)]
    """
    model = model.to(device)
    scores = []

    outer = tqdm(total=episodes, desc = 'Episodes', position=0)

    for trajectory in range(episodes):
        # get initial state
        observation, info = env.reset()
        observation = torch.from_numpy(observation).to(device)

        transitions = []
        action_probs = []

        for h in range(horizon):
            # sample a random action based on probabilities
            logits = model(observation)
            probs = torch.softmax(logits, dim=0)  # we have batch_dim
            dist = torch.distributions.Multinomial(total_count=1, probs=probs)

            # get action
            action_sample = dist.sample()
            action = torch.argmax(action_sample)

            # cache data for optimizations
            action_probs.append(probs)

            # interact with the environment. Store the old state
            prev_observation = observation

            observation, reward, done, truncated, info = env.step(action.item())
            observation = torch.from_numpy(observation).to(device)

            transitions.append((prev_observation, action, observation))

            # we ignore the reward for this environment cuz it's always 1.
            # rewards = ...

            if done or truncated:
                break


        # top is 500. it's the number of step taken in the environment.
        scores.append(len(transitions))
        rewards = torch.tensor([r for r in range(len(transitions), 0, -1)]).to(
            torch.float32
        )
        rewards = torch.tensor(rewards).to(device)

        # compute returns for every given step. (discounted sum of reward)
        returns = []
        for i in range(len(transitions)):
            r = 0
            power = 0
            for j in range(i, len(transitions)):
                r = r + ((gamma**power) * rewards[j]).numpy()
                power += 1
            returns.append(r)

        returns = torch.tensor(returns, device=device)

        # compute log_prob of selected action.
        # actions = torch.tensor(actions, device=device)
        actions = torch.tensor([a for (s, a, r) in transitions]).to(device)
        batch_probs = torch.stack(action_probs, dim=0)
        batch_probs = batch_probs.gather(
            dim=1, index=actions.long().view(-1, 1)
        ).squeeze()
        action_logprobs = torch.log(
            batch_probs
        )  # the log_prob of a multinomial is just the log of the probability of the action.

        # calculate the loss. the sign - is because works minimizing instead of maximizing
        loss = -torch.sum(action_logprobs * returns)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #update progress bar with debug info
        scores_num = len(scores)
        avg_score = np.mean(scores[np.max([-scores_num, -50]):-1])
        outer.set_description_str(f"Avg Score: {avg_score:.2f}")
        outer.update(1)

    return scores


scores = train(model, env, optimizer, device, horizon=500, episodes=500)


def running_mean(x, samples=50):
    kernel = np.ones(samples)
    conv_len = x.shape[0] - samples
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i : i + samples]
        y[i] /= samples
    return y


score = np.array(scores)
avg_score = running_mean(score, 50)
plt.figure(figsize=(15, 7))
plt.ylabel("Episode Length / Reward", fontsize=12)
plt.xlabel("Epochs", fontsize=12)
plt.plot(score, color="gray", linewidth=1)
plt.plot(avg_score, color="blue", linewidth=3)
plt.scatter(np.arange(score.shape[0]), score, color="green", linewidth=0.3)
plt.show()


def watch_agent():
    env = gym.make("CartPole-v1", render_mode="human")
    state, info = env.reset()
    rewards = []
    for t in range(2000):
        logits = model(torch.from_numpy(state).float())
        pred = torch.softmax(logits, dim=0)
        action = np.random.choice(np.array([0, 1]), p=pred.squeeze().data.numpy())
        state, reward, done, truncated, _ = env.step(action)
        rewards.append(reward)
        if done or truncated:
            print("Reward:", sum([r for r in rewards]))
            break
    env.close()


watch_agent()
