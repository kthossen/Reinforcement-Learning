#!/usr/bin/env python3
# dvar_deterministic.py
# dVaR: unweighted next-state order statistic k = ceil(α·S) - 1, using csv_mdp_loader.

import numpy as np, argparse, json
from csv_mdp_loader import load_all_domains, DOMAINS

def dvar_dp(P,R,gamma=0.9,T=100,J=4096):
    S,A = R.shape
    q_prev = np.zeros((S,J,A), dtype=np.float32)
    alphas = (np.arange(J, dtype=np.float32)/J)
    for _ in range(T):
        V_prev = q_prev.max(axis=2)   # (S,J)
        q_curr = np.empty_like(q_prev)
        orders = [np.argsort(V_prev[:,j]) for j in range(J)]
        V_sorted = [V_prev[orders[j], j] for j in range(J)]
        for s in range(S):
            for a in range(A):
                base = R[s,a]
                for j,alpha in enumerate(alphas):
                    vs = V_sorted[j]
                    if vs.size==0: q_curr[s,j,a]=base; continue
                    k = int(np.ceil(float(alpha)*vs.size)) - 1
                    if k<0: k=0
                    if k>=vs.size: k=vs.size-1
                    q_curr[s,j,a] = base + gamma*vs[k]
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
        qT = dvar_dp(P,R,gamma,T,J)
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
    ap.add_argument("--out", type=str, default="dvar_row.json")
    args=ap.parse_args()
    row = compute_row(args.data_dir,args.alpha,args.gamma,args.T,args.J)
    json.dump(row, open(args.out, "w"), indent=2)
    paper_print_row("dVaR", row, args.decimals)
