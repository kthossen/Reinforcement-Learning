
import numpy as np
from .risk import var

def value_iteration_var(mdp, alpha=0.25, n_quantiles=256, tol=1e-6, max_iter=10_000):
    """Quantile-based Value Iteration (VaR of return) by distributional Bellman backup.
    This is a simplified discretized approach:
    - Z(s) keeps a support of returns samples updated by Bellman operator using empirical sampling.
    - Policy is greedy w.r.t. VaR_alpha of action-values.
    Note: This is a faithful *approximation* to the Julia code goals, not a line-by-line port.
    """
    S, A = mdp.S, mdp.A
    rng = np.random.default_rng(0)
    # Initialize value distributions with zeros
    Z = np.zeros((S, n_quantiles), dtype=float)
    # Precompute quantile grid for reporting
    grid = np.linspace(0, 1, n_quantiles, endpoint=False) + 0.5/n_quantiles

    def q_value_samples(s, a, n_samples=1024):
        # Sample next states and compute one-step return + discounted next value sample
        probs = mdp.P[s, a]
        next_states = rng.choice(S, size=n_samples, p=probs)
        r = mdp.R[s, a]
        # bootstrap by sampling from Z(next_state)
        idx = rng.integers(0, n_quantiles, size=n_samples)
        boot = Z[next_states, idx]
        return r + mdp.gamma * boot

    for it in range(max_iter):
        delta = 0.0
        Z_new = np.empty_like(Z)
        for s in range(S):
            # Evaluate actions by VaR_alpha of sampled Q
            q_vars = []
            q_samples_actions = []
            for a in range(A):
                samples = q_value_samples(s, a)
                q_samples_actions.append(samples)
                q_vars.append(var(samples, alpha=alpha))
            a_star = int(np.argmax(q_vars))  # risk-averse: maximize lower-tail VaR
            # New distribution = the action's return distribution
            chosen = np.sort(q_samples_actions[a_star])
            # Downsample to n_quantiles via interpolation
            qs = np.quantile(chosen, grid, method="linear")
            Z_new[s] = qs
            delta = max(delta, float(np.max(np.abs(Z_new[s] - Z[s]))))
        Z = Z_new
        if delta < tol:
            break
    # Derive greedy policy
    policy = np.zeros(S, dtype=int)
    for s in range(S):
        q_vars = [var(q_value_samples(s, a), alpha=alpha) for a in range(A)]
        policy[s] = int(np.argmax(q_vars))
    # Compute state values as VaR_alpha of Z
    V = np.array([var(Z[s], alpha=alpha) for s in range(S)], dtype=float)
    return V, policy, Z
