"""
Deep Q-Network (DQN)
====================
Implements the DQN algorithm with:
  - Experience replay buffer
  - Target network (hard update every C steps)
  - ε-greedy exploration

Reference:
    Mnih et al., "Human-level control through deep reinforcement learning",
    Nature, 2015.  https://www.nature.com/articles/nature14236
"""

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN training loop
# ---------------------------------------------------------------------------
def dqn(env, num_episodes=500, hidden_dim=128, lr=1e-3, gamma=0.99,
        batch_size=64, buffer_capacity=10_000, target_update_freq=10,
        epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    """
    Train an agent with DQN.

    Parameters
    ----------
    env                : gym.Env
    num_episodes       : int
    hidden_dim         : int
    lr                 : float
    gamma              : float
    batch_size         : int
    buffer_capacity    : int
    target_update_freq : int, episodes between hard target-network updates
    epsilon_*          : exploration schedule

    Returns
    -------
    q_net          : trained QNetwork
    episode_rewards: list of float
    """
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    q_net = QNetwork(obs_dim, hidden_dim, act_dim)
    target_net = QNetwork(obs_dim, hidden_dim, act_dim)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(buffer_capacity)

    epsilon = epsilon_start
    episode_rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    action = int(q_net(state_t).argmax(dim=1).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward

            # Learning step
            if len(replay_buffer) >= batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = \
                    replay_buffer.sample(batch_size)

                with torch.no_grad():
                    max_next_q = target_net(next_states_b).max(dim=1).values
                    td_targets = rewards_b + gamma * max_next_q * (1 - dones_b)

                q_values = q_net(states_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)
                loss = criterion(q_values, td_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        # Hard update of target network
        if (ep + 1) % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        if (ep + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-50:])
            print(f"Episode {ep + 1}/{num_episodes}  avg reward (last 50): {avg:.2f}  ε={epsilon:.3f}")

    return q_net, episode_rewards


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    q_net, rewards = dqn(env, num_episodes=500)
    env.close()
    print(f"\nFinal avg reward (last 50 eps): {np.mean(rewards[-50:]):.2f}")
