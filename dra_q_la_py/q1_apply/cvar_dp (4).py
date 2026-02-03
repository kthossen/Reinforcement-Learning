#!/usr/bin/env python3
# cvar_dp.py
# CVaR-α DP of next-state mixture (weighted tail expectation), using csv_mdp_loader.

import numpy as np, argparse, json
from csv_mdp_loader import load_all_domains, DOMAINS

def weighted_cvar(values, weights, alpha):
    order = np.argsort(values)
    v = np.asarray(values, np.float32)[order]
    w = np.asarray(weights, np.float32)[order]
    tot = float(w.sum())
    if tot <= 0.0: return float(v[0]) if len(v) else 0.0
    w = w / tot
    cvar_mass = 0.0; cvar_val = 0.0; prev_cum = 0.0
    for i in range(len(v)):
        wi = float(w[i]); next_cum = prev_cum + wi
        take = min(wi, max(0.0, alpha - prev_cum))
        cvar_val += v[i] * take; cvar_mass += take
        prev_cum = next_cum
        if prev_cum >= alpha: break
    return float(cvar_val / cvar_mass) if cvar_mass > 0 else float(v[0])

def cvar_dp(P,R,gamma=0.9,T=100,J=4096):
    S,A = R.shape
    q_prev = np.zeros((S,J,A), dtype=np.float32)
    alphas = (np.arange(J, dtype=np.float32)/J); alphas[0]=1e-8
    for _ in range(T):
        V_prev = q_prev.max(axis=2)  # (S,J)
        q_curr = np.empty_like(q_prev)
        for s in range(S):
            for a in range(A):
                base = R[s,a]; w = P[s,a]
                for j,alpha in enumerate(alphas):
                    vals = base + gamma * V_prev[:, j]
                    q_curr[s,j,a] = weighted_cvar(vals, w, float(alpha))
        q_prev = q_curr
    return q_prev

def q_at_alpha(q_T, alpha):
    S,J,A = q_T.shape
    j = int(np.floor(alpha*J)); j = min(max(j,0), J-1)
    return q_T[:, j, :]

def compute_row(data_dir=".", alpha=0.25, gamma=0.9, T=100, J=4096):
    mdps = load_all_domains(data_dir)
    out={}
    for name in DOMAINS:
        P,R,s0 = mdps[name].P, mdps[name].R, mdps[name].S0
        qT = cvar_dp(P,R,gamma,T,J)
        out[name] = float(q_at_alpha(qT, alpha)[s0].max())
    return out

def paper_print_row(name,row,decimals=2):
    print("CW INV1 INV2 MR POP RS GR")
    order = ["CW","INV1","INV2","MR","POP","RS","GR"]
    print(f"{name} " + " ".join(format(row[d], f'.{decimals}f') for d in order))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=".")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--J", type=int, default=4096)
    ap.add_argument("--decimals", type=int, default=2)
    ap.add_argument("--out", type=str, default="cvar_row.json")
    args=ap.parse_args()
    row = compute_row(args.data_dir,args.alpha,args.gamma,args.T,args.J)
    json.dump(row, open(args.out, "w"), indent=2)
    paper_print_row("CVaR", row, args.decimals)
