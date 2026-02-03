#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np, pandas as pd
from pathlib import Path
import argparse, json

# --- Domains and local CSV mapping (your uploaded files) ---
DOMAIN_FILES = {
    "CW":   "data/csv/cliff.csv",
    "INV1": "data/csv/inventory1.csv",
    "INV2": "data/csv/inventory2.csv",
    "MR":   "data/csv/machine.csv",
    "POP":  "data/csv/population.csv",
    "RS":   "data/csv/riverswim.csv",
    "GR":   "data/csv/ruin.csv",
}
# Initial states (per paper Table 2)
S0_BY_DOMAIN = {"MR":1,"GR":5,"INV1":10,"INV2":20,"RS":9,"POP":44,"CW":37}

def load_domain_from_single_csv(path):
    """
    Expect combined schema:
    idstatefrom, idaction, idstateto, probability, reward
    (exactly what the user uploaded)
    """
    df = pd.read_csv(path)
    req = {"idstatefrom","idaction","idstateto","probability","reward"}
    if not req.issubset(df.columns):
        raise ValueError(f"CSV {path} missing columns; found {df.columns.tolist()}")

    S = int(max(df["idstatefrom"].max(), df["idstateto"].max()) + 1)
    A = int(df["idaction"].max() + 1)  # assumes contiguous 0..A-1

    P = np.zeros((S, A, S), float)
    R = np.zeros((S, A), float)

    # Fill transitions and average reward per (s,a) if repeated rows
    # Group by (s,a); transitions sum to 1 across s'
    for (s,a), g in df.groupby(["idstatefrom","idaction"]):
        s = int(s); a = int(a)
        # transitions
        probs = g.groupby("idstateto")["probability"].sum()
        for s_next, prob in probs.items():
            P[s, a, int(s_next)] = float(prob)
        # reward: mean of reward column for this (s,a)
        R[s, a] = float(g["reward"].mean())

    # Normalize safeguards
    row_sums = P.sum(axis=2, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    P = P / row_sums
    return P, R

def var_dp_discretized(P, R, gamma=0.9, T=100, J=512):
    """
    Uniform-discretized VaR operator (Example 4.4 / Eq. 12 in the paper).
    q[0]=0; q[t] = B_u^d q[t-1]. Return q_T (S,J,A).
    NOTE: J can be large (e.g., 4096) but computation grows ~O(S*A*S*J + S*A*J log(S*J)).
    """
    S, A = R.shape
    q_prev = np.zeros((S, J, A), float)
    j_w = np.full(J, 1.0/J)
    for _ in range(1, T+1):
        V_prev = q_prev.max(axis=2)  # (S,J)
        q_curr = np.empty_like(q_prev)
        for s in range(S):
            for a in range(A):
                y = R[s,a] + gamma * V_prev          # (S,J)
                w = P[s,a][:,None] * j_w[None,:]     # (S,J)
                y_flat = y.reshape(-1)
                w_flat = w.reshape(-1)
                order = np.argsort(y_flat)
                y_sorted = y_flat[order]
                w_sorted = w_flat[order]
                tot = w_sorted.sum()
                if tot <= 0:
                    # No mass -> degenerate at immediate reward
                    q_curr[s, :, a] = R[s, a]
                else:
                    cum_lt = np.concatenate(([0.0], np.cumsum(w_sorted[:-1]))) / tot
                    alphas = np.arange(J) / J
                    idxs = np.searchsorted(cum_lt, alphas, side="right") - 1
                    idxs = np.clip(idxs, 0, len(y_sorted)-1)
                    q_curr[s, :, a] = y_sorted[idxs]
        q_prev = q_curr
    return q_prev

def q_upper_at_alpha(q_T, alpha):
    S,J,A = q_T.shape
    j = min(int(np.ceil(alpha*J)), J-1)
    return q_T[:, j, :]  # (S,A)

def compute_qbar_row(domains, J=512, gamma=0.9, T=100, alpha0=0.25):
    out = {}
    for dom in domains:
        csv_path = DOMAIN_FILES[dom]
        P, R = load_domain_from_single_csv(csv_path)
        qT = var_dp_discretized(P, R, gamma=gamma, T=T, J=J)
        qbar = q_upper_at_alpha(qT, alpha0)     # (S,A)
        s0 = S0_BY_DOMAIN[dom]
        out[dom] = float(qbar[s0].max())
        print(f"{dom}: {out[dom]:.4f}")
    return out

def main():
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--J", type=int, default=32, help="risk grid size; use 4096 for paper-level fidelity")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--out", type=str, default="qbar_row.json")
    args = ap.parse_args()

    domains = ["CW","INV1","INV2","MR","POP","RS","GR"]
    row = compute_qbar_row(domains, J=args.J, gamma=args.gamma, T=args.T, alpha0=args.alpha)
    with open(args.out, "w") as f:
        json.dump(row, f, indent=2)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()

