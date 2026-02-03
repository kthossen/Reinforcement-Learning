
import numpy as np

class TabularMDP:
    """Finite MDP defined by P(s'|s,a) and rewards r(s,a).
    States are 0..S-1, actions 0..A-1.
    P shape: (S, A, S). R shape: (S, A).
    """
    def __init__(self, P, R, gamma=0.99, initial_state=0):
        P = np.asarray(P, dtype=float)
        R = np.asarray(R, dtype=float)
        assert P.ndim == 3 and R.ndim == 2
        S, A, Sp = P.shape
        assert Sp == S and R.shape == (S, A)
        self.S, self.A = S, A
        self.P = P
        self.R = R
        self.gamma = float(gamma)
        self.initial_state = int(initial_state)

    @staticmethod
    def from_csv(transitions_df, rewards_df, gamma=0.99, initial_state=0):
        """Build MDP from two CSV-like dataframes.
        transitions_df columns: s, a, s_next, prob
        rewards_df columns: s, a, reward
        """
        import pandas as pd  # local import
        S = int(max(transitions_df['s'].max(), transitions_df['s_next'].max()) + 1)
        A = int(max(transitions_df['a'].max(), rewards_df['a'].max()) + 1)
        P = np.zeros((S, A, S), dtype=float)
        for _, row in transitions_df.iterrows():
            P[int(row.s), int(row.a), int(row.s_next)] += float(row.prob)
        R = np.zeros((S, A), dtype=float)
        for _, row in rewards_df.iterrows():
            R[int(row.s), int(row.a)] = float(row.reward)
        # Normalize transitions to avoid numeric drift
        P /= P.sum(axis=2, keepdims=True).clip(min=1e-12)
        return TabularMDP(P, R, gamma=gamma, initial_state=initial_state)

    def step(self, s, a, rng):
        probs = self.P[s, a]
        s_next = rng.choice(self.S, p=probs)
        r = self.R[s, a]
        return s_next, r

    def rollout(self, policy, horizon=100, n_episodes=1, seed=0):
        rng = np.random.default_rng(seed)
        returns = []
        for ep in range(n_episodes):
            s = self.initial_state
            g = 0.0
            discount = 1.0
            for t in range(horizon):
                a = policy[s] if np.ndim(policy)==1 else np.argmax(policy[s])
                s, r = self.step(s, a, rng)
                g += discount * r
                discount *= self.gamma
            returns.append(g)
        return np.array(returns, dtype=float)
