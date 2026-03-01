"""
Actor-Critic (Advantage Actor-Critic, A2C — single worker)
===========================================================
Combines a policy gradient (actor) with a learned value function (critic).
The critic provides a baseline that reduces variance:

  δ = r + γ V(s') − V(s)          (TD error ≈ advantage)
  θ_actor  ← θ_actor  + α_π  · δ · ∇ log π(a|s)
  θ_critic ← θ_critic − α_V  · δ · ∇ V(s)

Reference:
    Sutton & Barto, Chapter 13.5.
    Mnih et al., "Asynchronous Methods for Deep RL (A3C)", ICML 2016.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1),
        )
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor_head(h), self.critic_head(h)


def actor_critic(env, num_episodes=1000, hidden_dim=128, lr=1e-3, gamma=0.99):
    """
    Train an agent with one-step Actor-Critic.

    Parameters
    ----------
    env         : gym.Env
    num_episodes: int
    hidden_dim  : int
    lr          : float
    gamma       : float

    Returns
    -------
    net            : trained ActorCriticNetwork
    episode_rewards: list of float
    """
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    net = ActorCriticNetwork(obs_dim, hidden_dim, act_dim)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    episode_rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            probs, value = net(state_t)

            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # TD error (advantage estimate)
            next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
            _, next_value = net(next_state_t)
            td_target = reward + gamma * next_value.detach() * (not done)
            advantage = td_target - value

            actor_loss = -dist.log_prob(action) * advantage.detach()
            critic_loss = advantage.pow(2)
            loss = actor_loss + critic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"Episode {ep + 1}/{num_episodes}  avg reward (last 100): {avg:.2f}")

    return net, episode_rewards


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    net, rewards = actor_critic(env, num_episodes=1000)
    env.close()
    print(f"\nFinal avg reward (last 100 eps): {np.mean(rewards[-100:]):.2f}")
