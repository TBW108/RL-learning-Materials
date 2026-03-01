"""
Proximal Policy Optimization (PPO) — Clip variant
==================================================
Collects a fixed number of environment steps per update, then performs
multiple epochs of minibatch gradient ascent on a clipped surrogate
objective to prevent large policy updates:

  L_CLIP(θ) = E[ min(r_t(θ) Â_t,  clip(r_t(θ), 1-ε, 1+ε) Â_t) ]

where r_t(θ) = π_θ(a|s) / π_θ_old(a|s).

Reference:
    Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017.
    https://arxiv.org/abs/1707.06347
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


# ---------------------------------------------------------------------------
# Actor-Critic network shared for PPO
# ---------------------------------------------------------------------------
class PPOActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_dim, output_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = self.actor_head(h)
        value = self.critic_head(h)
        return logits, value

    def act(self, state_t):
        logits, value = self.forward(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, states_t, actions_t):
        logits, values = self.forward(states_t)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


# ---------------------------------------------------------------------------
# PPO training loop
# ---------------------------------------------------------------------------
def ppo(env, total_timesteps=200_000, hidden_dim=64, lr=3e-4, gamma=0.99,
        gae_lambda=0.95, clip_eps=0.2, update_epochs=4, batch_size=64,
        rollout_steps=2048, entropy_coef=0.01, value_coef=0.5):
    """
    Train an agent with PPO (clip variant).

    Parameters
    ----------
    env             : gym.Env
    total_timesteps : int
    hidden_dim      : int
    lr              : float
    gamma           : float
    gae_lambda      : float, GAE-λ parameter
    clip_eps        : float, PPO clipping range ε
    update_epochs   : int, gradient epochs per rollout
    batch_size      : int, minibatch size for gradient updates
    rollout_steps   : int, steps collected before each update
    entropy_coef    : float, entropy bonus coefficient
    value_coef      : float, critic loss coefficient

    Returns
    -------
    net            : trained PPOActorCritic
    episode_rewards: list of float
    """
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    net = PPOActorCritic(obs_dim, hidden_dim, act_dim)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    episode_rewards, ep_reward = [], 0.0
    state, _ = env.reset()
    timestep = 0

    while timestep < total_timesteps:
        # ---- Collect rollout ----
        states, actions, log_probs_old, rewards, dones, values = [], [], [], [], [], []

        for _ in range(rollout_steps):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value = net.act(state_t)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state)
            actions.append(action.item())
            log_probs_old.append(log_prob.item())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())

            ep_reward += reward
            state = next_state
            timestep += 1

            if done:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                state, _ = env.reset()

        # ---- Compute GAE advantages ----
        with torch.no_grad():
            next_state_t = torch.FloatTensor(state).unsqueeze(0)
            _, last_value = net.forward(next_state_t)
            last_value = last_value.item()

        advantages, gae = [], 0.0
        for t in reversed(range(rollout_steps)):
            next_val = last_value if t == rollout_steps - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + torch.FloatTensor(values)

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        log_probs_old_t = torch.FloatTensor(log_probs_old)

        # ---- PPO update ----
        indices = np.arange(rollout_steps)
        for _ in range(update_epochs):
            np.random.shuffle(indices)
            for start in range(0, rollout_steps, batch_size):
                mb = indices[start:start + batch_size]
                log_probs_new, values_new, entropy = net.evaluate(states_t[mb], actions_t[mb])

                ratio = torch.exp(log_probs_new - log_probs_old_t[mb])
                adv_mb = advantages[mb]

                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (returns[mb] - values_new).pow(2).mean()
                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
                optimizer.step()

        if len(episode_rewards) > 0 and len(episode_rewards) % 10 == 0:
            avg = np.mean(episode_rewards[-10:])
            print(f"Timestep {timestep}/{total_timesteps}  "
                  f"episodes {len(episode_rewards)}  avg reward (last 10): {avg:.2f}")

    return net, episode_rewards


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    net, rewards = ppo(env, total_timesteps=200_000)
    env.close()
    if rewards:
        print(f"\nFinal avg reward (last 10 eps): {np.mean(rewards[-10:]):.2f}")
