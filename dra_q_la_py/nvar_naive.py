
#!/usr/bin/env python3
from csv_mdp_loader import load_domain_csv, DOMAIN_FILES, DOMAINS, S0_BY_DOMAIN
import numpy as np, argparse, json

def nvar_dp(P, R, gamma=0.9, T=100, J=256):
    S, A = R.shape
    q_prev = np.zeros((S, J, A), float)
    for _ in range(1, T+1):
        V_prev = q_prev.max(axis=2)  # (S,J)
        q_curr = np.empty_like(q_prev)
        for s in range(S):
            for a in range(A):
                y = R[s,a] + gamma * V_prev  # (S,J)
                q_curr[s, :, a] = (P[s,a][:,None] * y).sum(axis=0)
        q_prev = q_curr
    return q_prev

def q_at_alpha(q_T, alpha):
    S,J,A = q_T.shape
    j = int(np.floor(alpha*J)); j = np.clip(j, 0, J-1)
    return q_T[:, j, :]

def compute_row(alpha=0.25, T=100, J=256, gamma=0.9):
    out = {}
    for dom in DOMAINS:
        P, R = load_domain_csv(DOMAIN_FILES[dom])
        qT = nvar_dp(P, R, gamma=gamma, T=T, J=J)
        qslice = q_at_alpha(qT, alpha)
        s0 = S0_BY_DOMAIN[dom]
        out[dom] = float(qslice[s0].max())
        print(f"{dom}: {out[dom]:.2f}")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--J", type=int, default=128)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--out", type=str, default="nvar_row.json")
    args = ap.parse_args()
    row = compute_row(args.alpha, args.T, args.J, args.gamma)
    with open(args.out, "w") as f:
        json.dump(row, f, indent=2)
    print("Saved:", args.out)
