# Playground: Reinforcement Learning
Playground is a collection of single-file implementations of popular Reinforcement Learning algorithms, all written in Python and PyTorch. The goal of this repository is to make it easier for beginners to understand how RL algorithms work, by providing simple and easy-to-follow code examples.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
2.1 [Simple DQN](#simple-dqn)
3. [Contributing](#contributing)
4. [License](#license)

## Installation
To get started with Playground, you will first need to install Python 3.x and PyTorch. We recommend using Poetry to create and manage the virtual environment for this project. You can install Poetry by following the instructions on their website.

Once you have Poetry installed, simply run the following command to install the dependencies for Playground:

```
poetry install
```
This will create a virtual environment and install all the required dependencies for this project.

## Usage
Each implementation in Playground is contained in a single Python file. To run an implementation, simply execute the corresponding file. For example, to run the implementation of the DQN algorithm, you can use the following command:

```
python dqn.py --game=pong --training=true
```
The implementation will start running, and you should see the training progress and evaluation results in the console output.
You will also see the agent playing the game once it finished training.

#### Simple DQN
This is a simple implementation of the Deep Q-Network (DQN) algorithm in PyTorch. The code is provided as a single Python file, simple_dqn.py, which you can run on your local machine to train an agent on Gym environments games.

To run the simple_dqn.py script, use the following command:

```
python simple_dqn.py --tau <tau_value> --align-models-every <n_steps>
```

Modifying the values for <tau_value> and <n_steps> allows us to choose between using a soft or hard target network parameters update in the Deep Q-Network (DQN) algorithm.

Soft target network parameter update with a tau_value of 0.005 and updating the target network's weights every 500 steps
```
python simple_dqn.py --tau 0.005 --align-models-every 1
```

Hard target network parameter update with a tau_value of 1 (i.e., no soft update) and updating the target network's weights every 450 steps:
```
python simple_dqn.py --tau 1 --align-models-every 450
```

## Contributing
We welcome contributions to Playground from the community. If you would like to contribute, please fork this repository and submit a pull request with your changes.

## License
Playground is released under the Apache 2.0 License.
