
"""Train soft quantile Q-learning on a CSV domain folder.
Usage:
  python -m dra_q_la.experiments.train_qlearning --domain_dir ./path/to/domain --alpha 0.25 --kappa 1e-4
"""
import argparse, pandas as pd, numpy as np, json, os
from ..mdp import TabularMDP
from ..qlearning import soft_quantile_q_learning

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain_dir", type=str, required=False, default=None, help="Folder with transitions.csv, rewards.csv, domains_info.json")
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--kappa", type=float, default=1e-4)
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.domain_dir is None:
        # Use built-in toy
        from ..toy import load_toy_mdp
        mdp = load_toy_mdp()
    else:
        transitions = pd.read_csv(os.path.join(args.domain_dir, "transitions.csv"))
        rewards = pd.read_csv(os.path.join(args.domain_dir, "rewards.csv"))
        info = json.load(open(os.path.join(args.domain_dir, "domains_info.json"), "r"))
        mdp = TabularMDP.from_csv(transitions, rewards, gamma=info.get("gamma", 0.99), initial_state=info.get("initial_state", 0))

    Z, policy = soft_quantile_q_learning(mdp, alpha=args.alpha, kappa=args.kappa,
                                         episodes=args.episodes, horizon=args.horizon,
                                         lr=args.lr, epsilon=args.epsilon, seed=args.seed)
    out = {
        "policy": policy.tolist(),
        "alpha": args.alpha,
        "kappa": args.kappa,
    }
    os.makedirs("experiment/run/train/Q_out", exist_ok=True)
    with open("experiment/run/train/Q_out/policy.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved policy to experiment/run/train/Q_out/policy.json")

if __name__ == "__main__":
    main()
