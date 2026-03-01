# Algorithms

Python implementations of classic and modern Reinforcement Learning algorithms.

## Structure

```
Algorithms/
├── dynamic_programming/   # Value Iteration, Policy Iteration
├── model_free/            # Q-Learning, SARSA, TD(0)
├── policy_gradient/       # REINFORCE, Actor-Critic
└── deep_rl/               # DQN, PPO
```

## Algorithms Overview

### Dynamic Programming
Require a complete model of the environment (transition probabilities and rewards).

| Algorithm | File | Description |
|-----------|------|-------------|
| Value Iteration | [dynamic_programming/value_iteration.py](dynamic_programming/value_iteration.py) | Finds optimal value function by iteratively applying the Bellman optimality operator |
| Policy Iteration | [dynamic_programming/policy_iteration.py](dynamic_programming/policy_iteration.py) | Alternates between policy evaluation and policy improvement |

### Model-Free Methods
Learn directly from interaction with the environment without a model.

| Algorithm | File | Description |
|-----------|------|-------------|
| TD(0) | [model_free/td_zero.py](model_free/td_zero.py) | One-step temporal difference prediction |
| SARSA | [model_free/sarsa.py](model_free/sarsa.py) | On-policy TD control |
| Q-Learning | [model_free/q_learning.py](model_free/q_learning.py) | Off-policy TD control |

### Policy Gradient Methods
Directly optimize the policy using gradient ascent on expected return.

| Algorithm | File | Description |
|-----------|------|-------------|
| REINFORCE | [policy_gradient/reinforce.py](policy_gradient/reinforce.py) | Monte Carlo policy gradient |
| Actor-Critic | [policy_gradient/actor_critic.py](policy_gradient/actor_critic.py) | Combines policy gradient with value function baseline |

### Deep RL
Combines deep neural networks with RL algorithms.

| Algorithm | File | Description |
|-----------|------|-------------|
| DQN | [deep_rl/dqn.py](deep_rl/dqn.py) | Deep Q-Network with experience replay and target network |
| PPO | [deep_rl/ppo.py](deep_rl/ppo.py) | Proximal Policy Optimization |

## Running Examples

Each script can be run standalone. For example:

```bash
# Run Q-Learning on FrozenLake
python model_free/q_learning.py

# Run DQN on CartPole
python deep_rl/dqn.py
```

## Dependencies

```bash
pip install numpy gymnasium torch
```
