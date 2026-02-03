#!/usr/bin/env python3
# nvar_naive.py
# nVaR: expectation across next states on each α-slice, using csv_mdp_loader.

import numpy as np, argparse, json
from csv_mdp_loader import load_all_domains, DOMAINS

def nvar_dp(P,R,gamma=0.9,T=100,J=4096):
    S,A = R.shape
    q_prev = np.zeros((S,J,A), dtype=np.float32)
    for _ in range(T):
        V_prev = q_prev.max(axis=2)   # (S,J)
        q_curr = np.empty_like(q_prev)
        for s in range(S):
            for a in range(A):
                y = R[s,a] + gamma*V_prev  # (S,J)
                q_curr[s,:,a] = (P[s,a][:,None] * y).sum(axis=0)
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
        qT = nvar_dp(P,R,gamma,T,J)
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
    ap.add_argument("--out", type=str, default="nvar_row.json")
    args=ap.parse_args()
    row = compute_row(args.data_dir,args.alpha,args.gamma,args.T,args.J)
    json.dump(row, open(args.out, "w"), indent=2)
    paper_print_row("nVaR", row, args.decimals)
