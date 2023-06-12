import torch
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from torchinfo import summary
from tqdm import tqdm
from itertools import count
import matplotlib.pyplot as plt
import torch.nn.functional as F
import concurrent.futures
from tqdm import tqdm


# A2C (Advantage Actor-Critic) is a reinforcement learning algorithm that combines the benefits of both value-based and policy-based methods. 
# It learns a policy and a value function simultaneously and uses them to update the policy and estimate the action-value function.

# The policy defines the probability distribution over actions given the current state of the environment.
# The value function estimates the expected cumulative reward starting from the current state.

# The Policy 
# loss = -log π(a|s) * A(s, a)
# The policy gradient:
# ∇θ J(θ) ≈ 1/N ∑t=1,N ∇θ log π(a|s; θ) A(s, a)

# The value function loss:
# L = 1/2 ∑t=1,N (V(s_t) - r_t - γV(s_{t+1}))^2

# The total loss:
# L_total = loss + L + βH(π(s_t))

# H(π(s_t)) entropy of the policy distribution at state s_t
# β: entropy regularization coefficient.

# Make the gym environment 
env_name = 'CartPole-v1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define some hyperparameters
learning_rate = 2e-3
gamma = 0.99
agents_num = 8
steps_per_actor = 200
beta = 0.002
batch_size = 12
neurons = 256

max_steps = 500
epochs = 10000

class MLP(torch.nn.Module):
    def __init__(self, input_dim : int, output_dim: int, neurons : int) -> None:
        super().__init__()

        self.layer1 = torch.nn.Linear(input_dim, neurons)
        self.layer2 = torch.nn.Linear(neurons, neurons)
        self.layer3 = torch.nn.Linear(neurons, output_dim)
        self.critic = torch.nn.Linear(neurons, 1) # critic value

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return F.softmax(self.layer3(x), dim=1), self.critic(x)
    
    
class Agent():
    def __init__(self, env_name, model : torch.nn.Module, device = 'cpu'):
        self.env = gym.make(env_name)
        self.state, self.info = self.env.reset()
        self.total_steps = 0

        # init model.
        self.model = model.to(device)

        self.device = device

        self.actions = []
        self.probs = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        

    # Select action according to the policy
    def select_action(self, probs : torch.tensor):
        # Build a Categorical distribution (Multinomial) and returns the sampled action and it's log_prob
        categorical = torch.distributions.Categorical(probs=probs)
        action = categorical.sample()
        return action.to(self.device), categorical.log_prob(action).to(self.device)
    
    # Run model
    def run(self, obs):
        return self.model(obs)

    # Perform a training step, accumulating the gradients.
    # Train step will terminate after 'steps' or if the env returns 'terminated'
    def train_step(self, steps : int):
        done = False

        self.rewards = []
        self.actions = []
        self.probs = []
        self.log_probs = []
        self.values = []

        # collect data to perform training
        for i in range(steps):
            probs, value = self.run(torch.tensor(self.state).unsqueeze(0).to(self.device))
            action, log_prob = self.select_action(probs)

            self.state, rew, terminated, truncated, self.info = self.env.step(action.item())

            # store data
            self.probs.append(probs)
            self.rewards.append(rew)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.values.append(value)

            done = terminated or truncated
            if done:
                self.state, self.info = self.env.reset()
                break
        
    # compute returns (discounted)
    def compute_returns(self):
        ret = 0.0
        returns = []
        for t in reversed(range(len(self.rewards))):
            ret = ret + gamma * self.rewards[t]
            returns.append(ret)
        returns = [t for t in reversed(returns)]

        return returns
    

class Trainer():
    def __init__(self, env_name : str, concurrent_agent : int, neurons_num : int = 128, max_epoch_steps = 500, device = 'cpu') -> None:
        # create dummy env to get data to create a shared model.
        env = gym.make(env_name)
        state, _ = env.reset()

        # epoch steps
        self.max_epoch_steps = max_epoch_steps

        # init model.
        self.model = MLP(len(state), env.action_space.n, neurons_num).to(device)

        # init agents
        self.agents = [Agent(env_name, self.model, device) for _ in range(concurrent_agent)]
        self.device = device

        # init optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), learning_rate)
        

    def train(self, epochs : int):
        scores = []

        with concurrent.futures.ThreadPoolExecutor() as tpe:
            # collect data from the agents.
            for epoch in range(epochs):
                # update all the agents asynchronously and wait for them to terminate
                futures = [tpe.submit(agent.train_step, self.max_epoch_steps) for agent in self.agents]
                concurrent.futures.wait(futures)

                # get data
                score = 0
                self.optimizer.zero_grad()

                for agent in self.agents:
                    probs = torch.cat(agent.probs).to(self.device)
                    actions = torch.cat(agent.actions).to(self.device)
                    log_probs = torch.cat(agent.log_probs).to(self.device)
                    values = torch.cat(agent.values).squeeze().to(self.device)
                    returns = torch.tensor(agent.compute_returns()).to(self.device)

                    # compute advantage
                    advantages = returns - values

                    # LEARN
                    # losses
                    policy_loss = -(log_probs * advantages).sum()
                    critic_loss = 0.5 * advantages.square().sum()

                    # entropy
                    prob_of_action = probs.gather(1, actions.unsqueeze(-1))
                    entropy = -(log_probs * prob_of_action.squeeze()).sum()

                    # total loss
                    loss = policy_loss + critic_loss + beta * entropy
                    
                    # this call accumulates the gradients.
                    loss.backward()

                    score += np.sum(agent.rewards)


                # clip for stability
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
                self.optimizer.step()

                # store scores data
                scores.append(score / len(self.agents))

                mean = np.mean(scores[np.max([-len(scores), -50]):-1]) if len(scores) > 1 else scores[0]
                print(f"Epoch: {epoch+1} Reward {mean:.2f}")


                #self._update_agents_model()


if __name__ == "__main__":
    trainer = Trainer(env_name, agents_num, neurons, max_steps, device)
    trainer.train(epochs)
