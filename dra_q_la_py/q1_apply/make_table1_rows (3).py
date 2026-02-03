#!/usr/bin/env python3
# make_table1_rows.py
# Run all methods (paper settings) and print the table in one shot.

import json, subprocess, argparse
from pathlib import Path

order = ["CW","INV1","INV2","MR","POP","RS","GR"]
root = Path(__file__).parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=str(root), help="Where CSVs live")
    ap.add_argument("--alpha", type=str, default="0.25")
    ap.add_argument("--gamma", type=str, default="0.9")
    ap.add_argument("--T", type=str, default="100")
    ap.add_argument("--J", type=str, default="4096")
    args = ap.parse_args()

    cmds = [
        (["python", str(root/"algo1_static_var.py"), "--data_dir", args.data_dir, "--alpha",args.alpha,"--gamma",args.gamma,"--horizon","100","--n_rollouts","5000","--out", str(root/"algo1_row.json")], "Algorithm 1"),
        (["python", str(root/"qd_adaptive_alpha.py"), "--data_dir", args.data_dir, "--alpha",args.alpha,"--T",args.T,"--J",args.J,"--gamma",args.gamma,"--out", str(root/"qd_row.json")], "qd"),
        (["python", str(root/"nvar_naive.py"), "--data_dir", args.data_dir, "--alpha",args.alpha,"--T",args.T,"--J",args.J,"--gamma",args.gamma,"--out", str(root/"nvar_row.json")], "nVaR"),
        (["python", str(root/"dvar_deterministic.py"), "--data_dir", args.data_dir, "--alpha",args.alpha,"--T",args.T,"--J",args.J,"--gamma",args.gamma,"--out", str(root/"dvar_row.json")], "dVaR"),
        (["python", str(root/"e_expectation_dp.py"), "--data_dir", args.data_dir, "--gamma",args.gamma,"--T",args.T,"--out", str(root/"e_row.json")], "E"),
        (["python", str(root/"cvar_dp.py"), "--data_dir", args.data_dir, "--alpha",args.alpha,"--T",args.T,"--J",args.J,"--gamma",args.gamma,"--out", str(root/"cvar_row.json")], "CVaR"),
        (["python", str(root/"evar_dp.py"), "--data_dir", args.data_dir, "--alpha",args.alpha,"--T",args.T,"--J",args.J,"--gamma",args.gamma,"--out", str(root/"evar_row.json")], "EVaR"),
    ]

    print("CW INV1 INV2 MR POP RS GR")
    for cmd, name in cmds:
        subprocess.run(cmd, check=True)
        out_file = Path(cmd[-1])
        row = json.loads(out_file.read_text())
        print(name, " ".join(f"{row[k]:.2f}" for k in order))

if __name__ == "__main__":
    main()
