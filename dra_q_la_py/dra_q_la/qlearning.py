
import numpy as np
from .risk import var

def _softmax(x, temp=1.0):
    z = (x - np.max(x)) / max(temp, 1e-8)
    e = np.exp(z)
    return e / np.sum(e)

def soft_quantile_q_learning(mdp, alpha=0.25, kappa=1e-4, episodes=10_000, horizon=100, lr=0.1, epsilon=0.1, seed=0):
    """Soft Quantile Q-Learning.
    - Maintains per-(s,a) samples of return estimates and updates via bootstrapping.
    - kappa: entropy (softness) coefficient; kappa=0 reduces toward greedy.
    We approximate quantile Q-values using an empirical distribution of returns.
    """
    rng = np.random.default_rng(seed)
    S, A = mdp.S, mdp.A
    # Maintain N samples per state-action (start with zeros)
    N = 64
    Z = np.zeros((S, A, N), dtype=float)
    counts = np.zeros((S, A), dtype=int)
    policy = np.full(S, 0, dtype=int)

    for ep in range(episodes):
        s = mdp.initial_state
        # epsilon-greedy over VaR_alpha
        for t in range(horizon):
            if rng.random() < epsilon:
                a = rng.integers(0, A)
            else:
                q_vars = [var(Z[s, a], alpha=alpha) for a in range(A)]
                if kappa > 0:
                    probs = _softmax(np.array(q_vars), temp=max(1e-8, 1.0/kappa))
                    a = int(rng.choice(A, p=probs))
                else:
                    a = int(np.argmax(q_vars))
            s_next, r = mdp.step(s, a, rng)
            # bootstrap target by sampling from best action at s_next
            next_q_vars = [var(Z[s_next, ap], alpha=alpha) for ap in range(A)]
            if kappa > 0:
                probs_next = _softmax(np.array(next_q_vars), temp=max(1e-8, 1.0/kappa))
                a_next = int(rng.choice(A, p=probs_next))
            else:
                a_next = int(np.argmax(next_q_vars))
            boot_idx = rng.integers(0, N)
            target = r + mdp.gamma * Z[s_next, a_next, boot_idx]
            # SGD-like update on one sample (moving average)
            counts[s, a] += 1
            beta = lr
            Z[s, a, boot_idx] = (1 - beta) * Z[s, a, boot_idx] + beta * target
            s = s_next
        # refresh policy snapshot for monitoring
        for ss in range(S):
            q_vars = [var(Z[ss, a], alpha=alpha) for a in range(A)]
            policy[ss] = int(np.argmax(q_vars))
    return Z, policy

def evaluate_policy_rollout(mdp, policy, horizon=200, n_episodes=200, seed=123):
    returns = mdp.rollout(policy, horizon=horizon, n_episodes=n_episodes, seed=seed)
    return dict(mean=float(np.mean(returns)), std=float(np.std(returns)), 
                min=float(np.min(returns)), max=float(np.max(returns)),
                returns=returns.tolist())
