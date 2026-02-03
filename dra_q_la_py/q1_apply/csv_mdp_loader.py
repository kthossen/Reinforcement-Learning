#!/usr/bin/env python3
# csv_mdp_loader.py (offset-aware, with transition-level rewards R_sas)

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd
from typing import Dict, Tuple

DOMAINS = ["CW","INV1","INV2","MR","POP","RS","GR"]
FILES = {"CW":"cliff.csv","INV1":"inventory1.csv","INV2":"inventory2.csv","MR":"machine.csv",
         "POP":"population.csv","RS":"riverswim.csv","GR":"ruin.csv"}
# Paper initial states (often 1-based in CSVs). We'll auto-adjust to 0-based using CSV minima.
S0_BY_DOMAIN = {"MR":1,"GR":5,"INV1":10,"INV2":20,"RS":9,"POP":44,"CW":37}

@dataclass
class MDP:
    P: np.ndarray       # (S,A,S) transition probabilities
    R: np.ndarray       # (S,A) expected reward per (s,a)
    R_sas: np.ndarray   # (S,A,S) expected reward per transition (s,a,s')
    S0: int             # initial state (0-based after adjustment)
    name: str

def _normalize_rows(P: np.ndarray) -> np.ndarray:
    row_sums = P.sum(axis=2, keepdims=True)
    row_sums[row_sums==0.0] = 1.0
    return P / row_sums

def _ensure_self_loops(P: np.ndarray) -> None:
    S,A,_ = P.shape
    for s in range(S):
        for a in range(A):
            if P[s,a].sum() <= 0.0:
                P[s,a,s] = 1.0

def _read_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    if {"idstatefrom","idaction","idstateto","probability","reward"}.issubset(df.columns):
        cols = ("idstatefrom","idaction","idstateto","probability","reward")
    elif {"s","a","sp","p","r"}.issubset(df.columns):
        cols = ("s","a","sp","p","r")
    else:
        raise ValueError(f"{csv_path} missing required columns. Found: {df.columns.tolist()}")
    return df, cols

def load_domain_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str,int]]:
    df, (s_col,a_col,sp_col,p_col,r_col) = _read_csv(csv_path)

    # Auto-detect offsets and reindex to 0-based
    s_off  = int(df[s_col].min())
    sp_off = int(df[sp_col].min())
    a_off  = int(df[a_col].min())

    df = df.copy()
    df[s_col]  = df[s_col]  - s_off
    df[sp_col] = df[sp_col] - sp_off
    df[a_col]  = df[a_col]  - a_off

    S = int(max(df[s_col].max(), df[sp_col].max()) + 1)
    A = int(df[a_col].max() + 1)

    P = np.zeros((S,A,S), dtype=np.float32)
    R_sas = np.zeros((S,A,S), dtype=np.float32)

    # Aggregate per exact transition (s,a,sp)
    for (s,a,sp), g in df.groupby([s_col,a_col,sp_col]):
        s,a,sp = int(s), int(a), int(sp)
        P[s,a,sp] += float(g[p_col].sum())
        R_sas[s,a,sp] = float(g[r_col].mean())  # expected reward for that (s,a,sp)

    # Normalize transitions and ensure non-degenerate rows
    P = _normalize_rows(P)
    _ensure_self_loops(P)

    # Expected reward per (s,a) = sum_s' P * R_sas
    R = (P * R_sas).sum(axis=2).astype(np.float32)

    offsets = {"s_off": s_off, "sp_off": sp_off, "a_off": a_off}
    return P.astype(np.float32), R, R_sas.astype(np.float32), offsets

def load_all_domains(data_dir: Path | str) -> Dict[str, MDP]:
    data_dir = Path(data_dir)
    out: Dict[str, MDP] = {}
    for name in DOMAINS:
        csv_path = data_dir / FILES[name]
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected {csv_path} for domain {name}.")
        P, R, R_sas, offsets = load_domain_csv(csv_path)
        # Adjust initial state by the detected offset (clamp to valid range)
        s0_raw = S0_BY_DOMAIN[name]
        s0 = int(s0_raw - offsets["s_off"])
        s0 = max(0, min(P.shape[0]-1, s0))
        out[name] = MDP(P=P, R=R, R_sas=R_sas, S0=s0, name=name)
    return out

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=".")
    args = ap.parse_args()
    mdps = load_all_domains(args.data_dir)
    report = {n: {
        "S": int(m.P.shape[0]), "A": int(m.P.shape[1]), "S0": int(m.S0),
        "R_min": float(m.R.min()), "R_max": float(m.R.max()), "R_mean": float(m.R.mean()),
        "R_sas_min": float(m.R_sas.min()), "R_sas_max": float(m.R_sas.max())
    } for n,m in mdps.items()}
    print(json.dumps(report, indent=2))
