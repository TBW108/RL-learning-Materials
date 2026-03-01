"""
Q-Learning
==========
Off-policy TD control that directly learns the optimal action-value function
Q*(s, a) regardless of the policy being followed.

  Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') − Q(s, a)]

Reference:
    Watkins & Dayan, "Q-learning", Machine Learning, 1992.
    Sutton & Barto, Chapter 6.5.
"""

import numpy as np
import gymnasium as gym


def epsilon_greedy(Q, state, epsilon, num_actions):
    if np.random.random() < epsilon:
        return np.random.randint(num_actions)
    return int(np.argmax(Q[state]))


def q_learning(env, num_episodes=500, alpha=0.1, gamma=0.99,
               epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
    """
    Train an agent with Q-Learning.

    Parameters
    ----------
    env           : gym.Env with discrete observation & action spaces
    num_episodes  : int
    alpha         : float, learning rate
    gamma         : float, discount factor
    epsilon_start : float, initial exploration rate
    epsilon_end   : float, minimum exploration rate
    epsilon_decay : float, multiplicative decay per episode

    Returns
    -------
    Q              : np.ndarray, shape (S, A)
    episode_rewards: list of float
    """
    S = env.observation_space.n
    A = env.action_space.n
    Q = np.zeros((S, A))
    epsilon = epsilon_start
    episode_rewards = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = epsilon_greedy(Q, state, epsilon, A)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-Learning update (off-policy)
            td_target = reward + gamma * np.max(Q[next_state]) * (not done)
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state
            total_reward += reward

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"Episode {ep + 1}/{num_episodes}  avg reward (last 100): {avg:.3f}")

    return Q, episode_rewards


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)
    Q, rewards = q_learning(env, num_episodes=500)
    print("\nFinal Q-table:\n", Q)
    env.close()
