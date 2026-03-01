"""
Value Iteration
===============
Solves an MDP by iteratively applying the Bellman optimality operator until
the value function converges.

Reference:
    Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 4.
"""

import numpy as np


def value_iteration(num_states, num_actions, transition, reward, gamma=0.99, theta=1e-8):
    """
    Run value iteration to find the optimal value function and policy.

    Parameters
    ----------
    num_states  : int
    num_actions : int
    transition  : np.ndarray, shape (S, A, S) — transition probability P(s'|s,a)
    reward      : np.ndarray, shape (S, A)    — expected reward R(s,a)
    gamma       : float, discount factor in [0, 1)
    theta       : float, convergence threshold

    Returns
    -------
    V      : np.ndarray, shape (S,) — optimal state-value function
    policy : np.ndarray, shape (S,) — greedy policy with respect to V
    """
    V = np.zeros(num_states)

    while True:
        delta = 0.0
        for s in range(num_states):
            q_values = reward[s] + gamma * transition[s].dot(V)  # shape (A,)
            v_new = q_values.max()
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break

    policy = np.array([
        np.argmax(reward[s] + gamma * transition[s].dot(V))
        for s in range(num_states)
    ])
    return V, policy


# ---------------------------------------------------------------------------
# Demo: small random MDP
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    S, A = 5, 3

    # Random transition probabilities (rows sum to 1)
    T = np.random.dirichlet(np.ones(S), size=(S, A))
    R = np.random.randn(S, A)

    V_opt, pi_opt = value_iteration(S, A, T, R)
    print("Optimal value function:", V_opt)
    print("Optimal policy:        ", pi_opt)
