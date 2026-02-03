
# DRA-Q-LA (Python port, minimal)

A minimal, runnable Python port capturing the core ideas of the Julia repo
**Distributional Risk-Averse Quantile Q Learning (DRA-Q-LA)**.

This is not a line-by-line translation. It implements:
- Tabular MDP loading (from CSVs) and simulation
- Risk measures: mean, min, max, VaR, CVaR
- Quantile **Value Iteration** (VaR-based) using a discretized distributional backup
- **Soft Quantile Q-Learning** with kappa parameter
- Simple evaluation and a toy domain for smoke testing

## Install
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -e .
```

Or install dependencies only:
```bash
pip install -r requirements.txt
```

## Quick start (toy domain)
```bash
python -m dra_q_la.experiments.train_qlearning --alpha 0.25 --kappa 1e-4 --episodes 3000
python -m dra_q_la.experiments.eval_qlearning --use_dp --alpha 0.25
```

## CSV domain format
Expected files in `--domain_dir`:
- `transitions.csv` with columns: `s,a,s_next,prob`
- `rewards.csv` with columns: `s,a,reward`
- `domains_info.json` with keys: `{ "gamma": 0.99, "initial_state": 0 }`

## Notes
- The original Julia repo uses advanced constructs and multiple scripts (DP comparisons,
  multiple discretizations, Wasserstein distance plots). This port focuses on the core loop
  and APIs so you can train and evaluate quickly in Python, then extend as needed.
- Stochasticity means numerical results can differ from the Julia implementation.
