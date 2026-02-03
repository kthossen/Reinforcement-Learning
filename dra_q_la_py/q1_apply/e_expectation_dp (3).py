#!/usr/bin/env python3
# e_expectation_dp.py
# Standard discounted value iteration (expectation), using csv_mdp_loader.

import numpy as np, argparse, json
from csv_mdp_loader import load_all_domains, DOMAINS

def expectation_dp(P,R,gamma=0.9,T=100):
    S,A = R.shape
    V = np.zeros(S, dtype=np.float32)
    for _ in range(T):
        Q = np.empty((S,A), dtype=np.float32)
        for a in range(A):
            Q[:,a] = R[:,a] + gamma*(P[:,a,:] @ V)
        V = Q.max(axis=1)
    return V

def compute_row(data_dir=".", gamma=0.9, T=100):
    mdps = load_all_domains(data_dir)
    out={}
    for name in DOMAINS:
        P,R,s0 = mdps[name].P, mdps[name].R, mdps[name].S0
        V = expectation_dp(P,R,gamma,T)
        out[name] = float(V[s0])
    return out

def paper_print_row(name,row,decimals=2):
    print("CW INV1 INV2 MR POP RS GR")
    order = ["CW","INV1","INV2","MR","POP","RS","GR"]
    print(f"{name} " + " ".join(format(row[d], f'.{decimals}f') for d in order))

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=".")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--decimals", type=int, default=2)
    ap.add_argument("--out", type=str, default="e_row.json")
    args=ap.parse_args()
    row = compute_row(args.data_dir,args.gamma,args.T)
    json.dump(row, open(args.out, "w"), indent=2)
    paper_print_row("E", row, args.decimals)
