
# csv_mdp_loader.py
import numpy as np, pandas as pd
from dataclasses import dataclass


DOMAIN_FILES = {
    "CW":   "data/csv/cliff.csv",
    "INV1": "data/csv/inventory1.csv",
    "INV2": "data/csv/inventory2.csv",
    "MR":   "data/csv/machine.csv",
    "POP":  "data/csv/population.csv",
    "RS":   "data/csv/riverswim.csv",
    "GR":   "data/csv/ruin.csv",
}
S0_BY_DOMAIN = {"MR":1,"GR":5,"INV1":10,"INV2":20,"RS":9,"POP":44,"CW":37}
DOMAINS = ["CW","INV1","INV2","MR","POP","RS","GR"]

@dataclass
class TabularMDP:
    P: np.ndarray
    R: np.ndarray
    gamma: float = 0.9
    s0: int = 0
    @property
    def S(self): return self.P.shape[0]
    @property
    def A(self): return self.P.shape[1]

def load_domain_csv(path: str):
    df = pd.read_csv(path)
    req = {"idstatefrom","idaction","idstateto","probability","reward"}
    if not req.issubset(df.columns):
        raise ValueError(f"CSV {path} missing required columns {req}; found {df.columns.tolist()}")
    S = int(max(df["idstatefrom"].max(), df["idstateto"].max()) + 1)
    A = int(df["idaction"].max() + 1)
    P = np.zeros((S,A,S), float)
    R = np.zeros((S,A), float)
    for (s,a), g in df.groupby(["idstatefrom","idaction"]):
        s, a = int(s), int(a)
        probs = g.groupby("idstateto")["probability"].sum()
        for s_next, p in probs.items():
            P[s,a,int(s_next)] = float(p)
        R[s,a] = float(g["reward"].mean())
    row_sums = P.sum(axis=2, keepdims=True)
    row_sums[row_sums==0.0] = 1.0
    P = P / row_sums
    # ensure valid distributions; if a row is still zero, add self-loop
    S, A = P.shape[0], P.shape[1]
    for s in range(S):
        for a in range(A):
            if P[s,a].sum() <= 0:
                P[s,a,s] = 1.0
    return P, R
