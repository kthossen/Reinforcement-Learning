
"""Evaluate a saved Q-learning policy or run DP baseline and compare.
Usage:
  python -m dra_q_la.experiments.eval_qlearning [--domain_dir ...] [--use_dp]
"""
import argparse, pandas as pd, numpy as np, json, os
from ..mdp import TabularMDP
from ..qlearning import evaluate_policy_rollout, soft_quantile_q_learning
from ..quantile_dp import value_iteration_var

def load_mdp(domain_dir):
    if domain_dir is None:
        from ..toy import load_toy_mdp
        return load_toy_mdp()
    transitions = pd.read_csv(os.path.join(domain_dir, "transitions.csv"))
    rewards = pd.read_csv(os.path.join(domain_dir, "rewards.csv"))
    info = json.load(open(os.path.join(domain_dir, "domains_info.json"), "r"))
    return TabularMDP.from_csv(transitions, rewards, gamma=info.get("gamma", 0.99), initial_state=info.get("initial_state", 0))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain_dir", type=str, default=None)
    p.add_argument("--policy_path", type=str, default="experiment/run/train/Q_out/policy.json")
    p.add_argument("--use_dp", action="store_true", help="Also compute a VaR-DP policy for comparison.")
    p.add_argument("--alpha", type=float, default=0.25)
    args = p.parse_args()

    mdp = load_mdp(args.domain_dir)

    if os.path.exists(args.policy_path):
        pol = json.load(open(args.policy_path))["policy"]
        policy = np.array(pol, dtype=int)
    else:
        # train a quick policy if none exists
        Z, policy = soft_quantile_q_learning(mdp, alpha=args.alpha, episodes=2000, horizon=100, lr=0.1, epsilon=0.1, seed=0)

    res_q = evaluate_policy_rollout(mdp, policy, horizon=200, n_episodes=500, seed=123)

    print("Q-learning policy evaluation:")
    print(json.dumps(res_q, indent=2))

    if args.use_dp:
        V, pi_dp, Z = value_iteration_var(mdp, alpha=args.alpha, n_quantiles=256)
        res_dp = evaluate_policy_rollout(mdp, pi_dp, horizon=200, n_episodes=500, seed=321)
        print("\nVaR-DP policy evaluation:")
        print(json.dumps(res_dp, indent=2))

if __name__ == "__main__":
    main()
