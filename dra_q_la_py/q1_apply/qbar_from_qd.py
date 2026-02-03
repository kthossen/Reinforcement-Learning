#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# qbar_from_qd.py — compute q̄^d (empirical VaR) by rolling out the VaR–DP policy

import argparse, json
import numpy as np
from csv_mdp_loader import load_all_domains, DOMAINS  # uses your patched loader

def weighted_var_lower(values, weights, alpha):
    v = np.asarray(values, np.float64); w = np.asarray(weights, np.float64)
    tot = float(w.sum())
    if v.size == 0: return 0.0
    if tot <= 0.0: return float(v.min())
    w = w / tot
    alpha = float(max(min(alpha, 1.0), 1e-8))
    order = np.argsort(v)
    v, w = v[order], w[order]
    c = np.cumsum(w)
    idx = np.searchsorted(c, alpha, side="left")
    idx = min(max(int(idx), 0), v.size - 1)
    return float(v[idx])

def var_dp_qd(P, R_sas, gamma=0.9, T=100, J=4096):
    S, A, _ = P.shape
    q_prev = np.zeros((S, J, A), dtype=np.float64)
    alphas = (np.arange(J, dtype=np.float64) / J)
    if J > 0: alphas[0] = 1e-8
    for _ in range(T):
        V_prev = q_prev.max(axis=2)  # (S,J)
        q_curr = np.empty_like(q_prev)
        for s in range(S):
            for a in range(A):
                w = P[s, a]
                base = R_sas[s, a]
                for j, alpha in enumerate(alphas):
                    vals = base + gamma * V_prev[:, j]
                    q_curr[s, j, a] = weighted_var_lower(vals, w, float(alpha))
        q_prev = q_curr
    return q_prev  # shape (S,J,A)

def greedy_policy_from_q(q_T, alpha):
    """Return a state->action policy greedy w.r.t. q_T at given alpha index."""
    S, J, A = q_T.shape
    j = int(np.floor(alpha * J)); j = min(max(j, 0), J-1)
    Q = q_T[:, j, :]  # (S,A)
    return np.argmax(Q, axis=1), j  # (S,), j

def rollout_policy(P, R_sas, policy, gamma, s0, horizon, n_episodes=20000, seed=123):
    rng = np.random.default_rng(seed)
    S = P.shape[0]
    rets = np.empty(n_episodes, dtype=np.float64)
    for ep in range(n_episodes):
        s = s0
        disc = 1.0
        G = 0.0
        for t in range(horizon):
            a = int(policy[s])
            # safe fallback if row is degenerate
            p_row = P[s, a]
            if p_row.sum() <= 0:
                sp = s
                r = 0.0
            else:
                sp = int(rng.choice(S, p=p_row))
                r = float(R_sas[s, a, sp])
            G += disc * r
            disc *= gamma
            s = sp
        rets[ep] = G
    return rets

def compute_qbar_row(data_dir=".", alpha=0.25, gamma=0.9, T=100, J=4096,
                     horizon=100, n_episodes=20000, seed=123, decimals=2):
    mdps = load_all_domains(data_dir)
    print("CW INV1 INV2 MR POP RS GR")
    out = {}
    for name in DOMAINS:
        m = mdps[name]
        qT = var_dp_qd(m.P, m.R_sas, gamma=gamma, T=T, J=J)
        policy, j = greedy_policy_from_q(qT, alpha)
        rets = rollout_policy(m.P, m.R_sas, policy, gamma, m.S0, horizon,
                              n_episodes=n_episodes, seed=seed)
        qbar = float(np.quantile(rets, alpha))
        out[name] = qbar
    print("q\u0304d " + " ".join(f"{out[k]:.{decimals}f}" for k in ["CW","INV1","INV2","MR","POP","RS","GR"]))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--J", type=int, default=4096)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--n_episodes", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--decimals", type=int, default=2)
    ap.add_argument("--out", type=str, default="qbard_row.json")
    args = ap.parse_args()

    row = compute_qbar_row(args.data_dir, args.alpha, args.gamma, args.T, args.J,
                           args.horizon, args.n_episodes, args.seed, args.decimals)
    with open(args.out, "w") as f:
        json.dump(row, f, indent=2)

