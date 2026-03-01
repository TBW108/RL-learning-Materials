"""
Policy Iteration
================
Alternates between:
  1. Policy Evaluation  — compute V^π for the current policy π
  2. Policy Improvement — greedify π with respect to V^π

until the policy is stable (no change on improvement step).

Reference:
    Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 4.
"""

import numpy as np


def policy_evaluation(policy, num_states, transition, reward, gamma=0.99, theta=1e-8):
    """Iterative policy evaluation: solve V^π by repeated Bellman backups."""
    V = np.zeros(num_states)
    while True:
        delta = 0.0
        for s in range(num_states):
            a = policy[s]
            v_new = reward[s, a] + gamma * transition[s, a].dot(V)
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    return V


def policy_improvement(V, num_states, num_actions, transition, reward, gamma=0.99):
    """Return the greedy policy with respect to V."""
    return np.array([
        np.argmax(reward[s] + gamma * transition[s].dot(V))
        for s in range(num_states)
    ])


def policy_iteration(num_states, num_actions, transition, reward, gamma=0.99, theta=1e-8):
    """
    Run policy iteration to find the optimal policy.

    Parameters
    ----------
    num_states  : int
    num_actions : int
    transition  : np.ndarray, shape (S, A, S)
    reward      : np.ndarray, shape (S, A)
    gamma       : float
    theta       : float

    Returns
    -------
    V      : np.ndarray, shape (S,)
    policy : np.ndarray, shape (S,)
    """
    policy = np.zeros(num_states, dtype=int)

    while True:
        V = policy_evaluation(policy, num_states, transition, reward, gamma, theta)
        new_policy = policy_improvement(V, num_states, num_actions, transition, reward, gamma)
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy

    return V, policy


# ---------------------------------------------------------------------------
# Demo: small random MDP
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    S, A = 5, 3

    T = np.random.dirichlet(np.ones(S), size=(S, A))
    R = np.random.randn(S, A)

    V_opt, pi_opt = policy_iteration(S, A, T, R)
    print("Optimal value function:", V_opt)
    print("Optimal policy:        ", pi_opt)
