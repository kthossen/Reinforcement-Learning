
#!/usr/bin/env python3
from csv_mdp_loader import TabularMDP, load_domain_csv, DOMAIN_FILES, DOMAINS, S0_BY_DOMAIN
import numpy as np, argparse, json

def VaR(samples, alpha):
    x = np.asarray(samples, float)
    return float(np.quantile(x, alpha))

def baseline_mean_greedy(mdp: TabularMDP, s, rng):
    return int(np.argmax(mdp.R[s]))

def rollout_first_action(mdp: TabularMDP, a0, horizon, seed, baseline='mean_greedy'):
    rng = np.random.default_rng(seed)
    s = mdp.s0
    disc = 1.0
    G = 0.0
    s_next = rng.choice(mdp.S, p=mdp.P[s, a0])
    r = mdp.R[s, a0]; G += disc*r; disc *= mdp.gamma; s = s_next
    for t in range(1, horizon):
        a = baseline_mean_greedy(mdp, s, rng) if baseline=='mean_greedy' else int(rng.integers(0, mdp.A))
        s_next = rng.choice(mdp.S, p=mdp.P[s, a])
        r = mdp.R[s, a]; G += disc*r; disc *= mdp.gamma; s = s_next
    return G

def build_static_var_policy(mdp: TabularMDP, alpha, horizon, n_rollouts, seed=0):
    rng = np.random.default_rng(seed)
    Qv = np.zeros((mdp.S, mdp.A), float)
    s = mdp.s0
    for a in range(mdp.A):
        rets = [rollout_first_action(mdp, a, horizon, int(rng.integers(0, 2**31-1))) for _ in range(n_rollouts)]
        Qv[s, a] = VaR(rets, alpha)
    pi = np.argmax(Qv[s])
    return Qv, pi

def evaluate_policy(mdp: TabularMDP, pi, horizon, n_episodes=2000, seed=123):
    rng = np.random.default_rng(seed)
    rets = []
    for _ in range(n_episodes):
        s = mdp.s0; disc=1.0; G=0.0
        for t in range(horizon):
            a = int(pi) if t==0 else int(np.argmax(mdp.R[s]))
            s_next = rng.choice(mdp.S, p=mdp.P[s, a])
            r = mdp.R[s, a]
            G += disc*r; disc *= mdp.gamma; s = s_next
        rets.append(G)
    return np.asarray(rets, float)

def compute_row(alpha=0.25, horizon=100, n_rollouts=1000, gamma=0.9):
    out = {}
    for dom in DOMAINS:
        P, R = load_domain_csv(DOMAIN_FILES[dom])
        mdp = TabularMDP(P, R, gamma=gamma, s0=S0_BY_DOMAIN[dom])
        Qv, pi = build_static_var_policy(mdp, alpha, horizon, n_rollouts)
        rets = evaluate_policy(mdp, pi, horizon, n_episodes=max(1000, n_rollouts//2))
        out[dom] = float(np.quantile(rets, alpha))
        print(f"{dom}: {out[dom]:.2f}")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--n_rollouts", type=int, default=1000)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--out", type=str, default="algo1_row.json")
    args = ap.parse_args()
    row = compute_row(args.alpha, args.horizon, args.n_rollouts, args.gamma)
    with open(args.out, "w") as f:
        json.dump(row, f, indent=2)
    print("Saved:", args.out)
