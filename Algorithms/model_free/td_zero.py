"""
TD(0) — One-step Temporal Difference Prediction
================================================
Estimates the state-value function V^π for a fixed policy π using
one-step TD updates:

  V(s) ← V(s) + α [r + γ V(s') − V(s)]

Reference:
    Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 6.1.
"""

import numpy as np
import gymnasium as gym


def td_zero(env, policy, num_episodes=500, alpha=0.01, gamma=0.99):
    """
    Estimate V^π using TD(0).

    Parameters
    ----------
    env         : gym.Env with a discrete observation space
    policy      : callable(state) -> action
    num_episodes: int
    alpha       : float, learning rate
    gamma       : float, discount factor

    Returns
    -------
    V : np.ndarray, shape (S,) — estimated state-value function
    """
    S = env.observation_space.n
    V = np.zeros(S)

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            td_error = reward + gamma * V[next_state] * (not done) - V[state]
            V[state] += alpha * td_error
            state = next_state

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep + 1}/{num_episodes} done.")

    return V


# ---------------------------------------------------------------------------
# Demo: evaluate the uniform-random policy on FrozenLake
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False)
    A = env.action_space.n
    random_policy = lambda s: np.random.randint(A)

    V = td_zero(env, random_policy, num_episodes=500)
    print("\nEstimated V (random policy):\n", V.reshape(4, 4))
    env.close()
