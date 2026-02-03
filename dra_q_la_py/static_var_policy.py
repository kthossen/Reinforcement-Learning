
#!/usr/bin/env python3
"""
Algorithm 1: Static VaR Policy Execution (self-contained demo)

Summary
-------
Given a tabular MDP and a quantile level alpha, this algorithm builds a *static*
policy that maximizes the *Value-at-Risk* (VaR) of discounted returns, per state,
using offline Monte Carlo estimates. At execution time, the policy is fixed (static).

Steps
-----
1) For each state s and action a:
   - Simulate many rollouts that take action a at t=0, then follow a baseline policy.
   - Record the discounted return G per rollout.
   - Compute VaR_alpha over the set of G values => Q_VaR[s, a].
2) Static policy: pi[s] = argmax_a Q_VaR[s, a].
3) Execute pi; it remains fixed (no updates).

CLI
---
# Run the toy demo
python static_var_policy.py --alpha 0.25 --horizon 200 --n-rollouts 1000 --baseline mean_greedy

# Load a CSV domain (optional)
python static_var_policy.py --domain-dir /path/to/domain --alpha 0.25 --horizon 200 --n-rollouts 2000

Expected CSV domain format:
- transitions.csv with columns: s,a,s_next,prob
- rewards.csv with columns: s,a,reward
- domains_info.json with keys: { "gamma": 0.99, "initial_state": 0 }

Outputs
-------
- policy.json: {"alpha": ..., "policy": [...], "Q_VaR": [[...], ...]}
- summary_row.csv: one CSV row labeled "Algorithm 1" with CW, INV1, INV2, MR, POP, RS, GR
"""

import argparse, json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------- Tabular MDP --------------------
@dataclass
class TabularMDP:
    P: np.ndarray   # shape (S, A, S)
    R: np.ndarray   # shape (S, A)
    gamma: float = 0.99
    initial_state: int = 0

    @property
    def S(self): return self.P.shape[0]
    @property
    def A(self): return self.P.shape[1]

    @staticmethod
    def from_csv(domain_dir: Path, gamma=0.99, initial_state=0):
        transitions = pd.read_csv(domain_dir/"transitions.csv")
        rewards = pd.read_csv(domain_dir/"rewards.csv")
        S = int(max(transitions['s'].max(), transitions['s_next'].max()) + 1)
        A = int(max(transitions['a'].max(), rewards['a'].max()) + 1)
        P = np.zeros((S, A, S), float)
        for _, row in transitions.iterrows():
            P[int(row.s), int(row.a), int(row.s_next)] += float(row.prob)
        P /= P.sum(axis=2, keepdims=True).clip(min=1e-12)
        R = np.zeros((S, A), float)
        for _, row in rewards.iterrows():
            R[int(row.s), int(row.a)] = float(row.reward)
        return TabularMDP(P, R, gamma=float(gamma), initial_state=int(initial_state))

    def step(self, s, a, rng):
        s_next = rng.choice(self.S, p=self.P[s, a])
        r = self.R[s, a]
        return s_next, r

# -------------------- Toy domain --------------------
def load_toy_mdp():
    S, A = 3, 2
    P = np.zeros((S, A, S), float)
    for s in range(S):
        P[s,0,(s+1)%S] = 1.0
    for s in range(S):
        P[s,1,s] = 0.7
        P[s,1,(s+1)%S] = 0.3
    R = np.array([[0.0,-0.1],[0.0,0.0],[1.0,0.2]], float)
    gamma=0.95
    return TabularMDP(P, R, gamma=gamma, initial_state=0)

# -------------------- Risk helpers --------------------
def VaR(samples, alpha=0.25):
    x = np.asarray(samples, float)
    return float(np.quantile(x, alpha))

def CVaR(samples, alpha=0.25):
    x = np.asarray(samples, float)
    q = np.quantile(x, alpha)
    mask = x <= q
    return float(x[mask].mean() if np.any(mask) else q)

def probability_of_profit(x):
    x = np.asarray(x, float)
    return float(np.mean(x > 0))

def downside_deviation(x, mar=0.0):
    x = np.asarray(x, float)
    d = np.minimum(0.0, x - mar)
    return float(np.sqrt(np.mean(d**2)))

def geometric_growth(x):
    y = np.clip(1.0 + np.asarray(x, float), 1e-6, None)
    return float(np.exp(np.mean(np.log(y))) - 1.0)

# -------------------- Baseline policies --------------------
def policy_random(mdp: TabularMDP, s, rng):
    return int(rng.integers(0, mdp.A))

def policy_mean_greedy(mdp: TabularMDP, s, rng=None):
    # Greedy by immediate reward mean R[s,a]
    return int(np.argmax(mdp.R[s]))

BASELINES = {
    "random": policy_random,
    "mean_greedy": policy_mean_greedy,
}

# -------------------- Monte Carlo estimators --------------------
def rollout_with_first_action(mdp: TabularMDP, s0, a0, baseline_name="mean_greedy",
                              horizon=200, seed=0):
    rng = np.random.default_rng(seed)
    baseline = BASELINES[baseline_name]
    s = s0
    G = 0.0
    disc = 1.0

    # first action is fixed
    s, r = mdp.step(s, a0, rng)
    G += disc * r
    disc *= mdp.gamma

    # then follow baseline
    for t in range(1, horizon):
        a = baseline(mdp, s, rng)
        s, r = mdp.step(s, a, rng)
        G += disc * r
        disc *= mdp.gamma
    return G

def estimate_Q_VaR(mdp: TabularMDP, alpha=0.25, n_rollouts=2000, horizon=200,
                   baseline_name="mean_greedy", seed=0):
    rng = np.random.default_rng(seed)
    Q_VaR = np.zeros((mdp.S, mdp.A), float)
    for s in range(mdp.S):
        for a in range(mdp.A):
            rets = [rollout_with_first_action(mdp, s, a, baseline_name, horizon, seed=int(rng.integers(0, 2**31-1)))
                    for _ in range(n_rollouts)]
            Q_VaR[s, a] = VaR(rets, alpha)
    return Q_VaR

def build_static_var_policy(mdp: TabularMDP, alpha=0.25, n_rollouts=2000, horizon=200,
                            baseline_name="mean_greedy", seed=0):
    Qv = estimate_Q_VaR(mdp, alpha=alpha, n_rollouts=n_rollouts, horizon=horizon,
                        baseline_name=baseline_name, seed=seed)
    pi = np.argmax(Qv, axis=1).astype(int)
    return Qv, pi

def evaluate_policy(mdp: TabularMDP, policy, horizon=200, n_episodes=2000, seed=123):
    rng = np.random.default_rng(seed)
    rets = []
    for ep in range(n_episodes):
        s = mdp.initial_state
        disc = 1.0
        G = 0.0
        for t in range(horizon):
            a = int(policy[s])
            s, r = mdp.step(s, a, rng)
            G += disc * r
            disc *= mdp.gamma
        rets.append(G)
    rets = np.asarray(rets, float)
    return {
        "mean": float(np.mean(rets)),
        "std": float(np.std(rets)),
        "min": float(np.min(rets)),
        "max": float(np.max(rets)),
        "returns": rets,
    }

def summarize_row_from_returns(returns, alpha=0.25):
    r = np.asarray(returns, float)
    mu = float(np.mean(r))
    sd = float(np.std(r))
    # Placeholders for INV1/INV2/CW/MR/etc. Replace with your exact definitions as needed.
    CW   = VaR(r, alpha)            # VaR as a cost-like proxy
    INV1 = mu + sd                  # placeholder
    INV2 = mu + 2*sd                # placeholder
    MR   = mu                       # mean return
    POP  = probability_of_profit(r)
    RS   = downside_deviation(r)    # placeholder risk score
    GR   = geometric_growth(r)
    return dict(CW=CW, INV1=INV1, INV2=INV2, MR=MR, POP=POP, RS=RS, GR=GR)

# -------------------- CLI --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain-dir", type=str, default=None, help="Folder with transitions.csv, rewards.csv, domains_info.json")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--n-rollouts", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--baseline", type=str, default="mean_greedy", choices=list(BASELINES.keys()))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="algorithm1_static_var_out")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if args.domain_dir:
        # Load CSV domain
        info_path = Path(args.domain_dir)/"domains_info.json"
        gamma, s0 = 0.99, 0
        if info_path.exists():
            meta = json.load(open(info_path, "r"))
            gamma = float(meta.get("gamma", gamma)); s0 = int(meta.get("initial_state", s0))
        mdp = TabularMDP.from_csv(Path(args.domain_dir), gamma=gamma, initial_state=s0)
    else:
        mdp = load_toy_mdp()

    Qv, pi = build_static_var_policy(mdp, alpha=args.alpha, n_rollouts=args.n_rollouts,
                                     horizon=args.horizon, baseline_name=args.baseline, seed=args.seed)

    # Save policy
    with open(outdir/"policy.json", "w") as f:
        json.dump({"alpha": args.alpha, "policy": pi.tolist(), "Q_VaR": Qv.tolist()}, f, indent=2)

    # Evaluate and summarize as "Algorithm 1" row
    eval_res = evaluate_policy(mdp, pi, horizon=args.horizon, n_episodes=max(500, args.n_rollouts//2), seed=123)
    row = summarize_row_from_returns(eval_res["returns"], alpha=args.alpha)
    row_df = pd.DataFrame([row], index=["Algorithm 1"], columns=["CW","INV1","INV2","MR","POP","RS","GR"])
    row_df.to_csv(outdir/"summary_row.csv", float_format="%.4f")

    print("Saved ->", (outdir/"policy.json").resolve())
    print("Saved ->", (outdir/"summary_row.csv").resolve())

if __name__ == "__main__":
    main()
