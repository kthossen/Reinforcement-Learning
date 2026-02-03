
#!/usr/bin/env python3
from csv_mdp_loader import load_domain_csv, DOMAIN_FILES, DOMAINS, S0_BY_DOMAIN
import numpy as np, argparse, json

def var_dp_qd(P, R, gamma=0.9, T=100, J=64):
    S, A = R.shape
    q_prev = np.zeros((S, J, A), float)
    alphas = np.arange(J) / J
    for _ in range(1, T+1):
        V_prev = q_prev.max(axis=2)  # (S,J)
        q_curr = np.empty_like(q_prev)
        # Precompute sort orders of V_prev per j (same for all (s,a) up to a constant shift)
        orders = [np.argsort(V_prev[:, j]) for j in range(J)]
        V_sorted = [V_prev[orders[j], j] for j in range(J)]
        for a in range(A):
            # For each action, we can reuse R[:,a] shifts
            r_a = R[:, a]  # (S,)
            for s in range(S):
                w = P[s, a]  # (S,)
                for j, alpha in enumerate(alphas):
                    order = orders[j]
                    vs = V_sorted[j]  # sorted V_prev[:, j]
                    if vs.size == 0:
                        q_curr[s, j, a] = R[s, a]
                        continue
                    ws = w[order]
                    tot = float(ws.sum())
                    if tot <= 0:
                        q_curr[s, j, a] = R[s, a]
                        continue
                    cum_lt = np.concatenate(([0.0], np.cumsum(ws[:-1]))) / max(tot, 1e-12)
                    idx = np.searchsorted(cum_lt, alpha, side="right") - 1
                    if idx < 0: idx = 0
                    if idx >= len(vs): idx = len(vs)-1
                    q_curr[s, j, a] = r_a[s] + gamma * float(vs[idx])
        q_prev = q_curr
    return q_prev

def q_lower_at_alpha(q_T, alpha):
    S,J,A = q_T.shape
    j = int(np.floor(alpha*J)); j = np.clip(j, 0, J-1)
    return q_T[:, j, :]

def compute_row(alpha=0.25, T=100, J=64, gamma=0.9):
    out = {}
    for dom in DOMAINS:
        P, R = load_domain_csv(DOMAIN_FILES[dom])
        qT = var_dp_qd(P, R, gamma=gamma, T=T, J=J)
        qlow = q_lower_at_alpha(qT, alpha)
        s0 = S0_BY_DOMAIN[dom]
        out[dom] = float(qlow[s0].max())
        print(f"{dom}: {out[dom]:.2f}")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--T", type=int, default=60)
    ap.add_argument("--J", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--out", type=str, default="qd_row.json")
    args = ap.parse_args()
    row = compute_row(args.alpha, args.T, args.J, args.gamma)
    with open(args.out, "w") as f:
        json.dump(row, f, indent=2)
    print("Saved:", args.out)
