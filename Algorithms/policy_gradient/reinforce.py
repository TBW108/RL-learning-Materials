"""
REINFORCE — Monte Carlo Policy Gradient
========================================
Updates the policy network parameters θ in the direction that increases
expected return using complete episode returns:

  θ ← θ + α ∇_θ log π_θ(a|s) · G_t

Reference:
    Williams, "Simple Statistical Gradient-Following Algorithms for
    Connectionist Reinforcement Learning", Machine Learning, 1992.
    Sutton & Barto, Chapter 13.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


def reinforce(env, num_episodes=1000, hidden_dim=128, lr=1e-3, gamma=0.99):
    """
    Train a policy with REINFORCE.

    Parameters
    ----------
    env         : gym.Env
    num_episodes: int
    hidden_dim  : int
    lr          : float, learning rate
    gamma       : float, discount factor

    Returns
    -------
    policy_net     : trained PolicyNetwork
    episode_rewards: list of float
    """
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy_net = PolicyNetwork(obs_dim, hidden_dim, act_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    episode_rewards = []

    for ep in range(num_episodes):
        log_probs, rewards = [], []
        state, _ = env.reset()
        done = False

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            probs = policy_net(state_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            rewards.append(reward)

        # Compute discounted returns
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalize

        # Policy gradient update
        loss = -torch.stack([lp * Gt for lp, Gt in zip(log_probs, returns)]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(sum(rewards))
        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"Episode {ep + 1}/{num_episodes}  avg reward (last 100): {avg:.2f}")

    return policy_net, episode_rewards


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    policy_net, rewards = reinforce(env, num_episodes=1000)
    env.close()
    print(f"\nFinal avg reward (last 100 eps): {np.mean(rewards[-100:]):.2f}")
