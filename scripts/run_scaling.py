#!/usr/bin/env python3
"""Scaling study across qubit counts + realistic-noise comparison.

    python scripts/run_scaling.py --qubits 4 8 12 16 --seeds 3
    python scripts/run_scaling.py --quick

For each qubit count we run the QRC across seeds and report AUPRC/EDS mean +/- CI,
so the "expressivity grows with Hilbert dimension" claim carries error bars
instead of resting on single points. --noise-samples>0 also runs the PennyLane
density-matrix path (real depolarizing + amplitude damping) on a subset to test
the noise-as-regularizer hypothesis honestly.
"""

import os, sys, json, argparse
from dataclasses import replace
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from src.experiment import prepare_data, run_single
from src import metrics as M

RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RESULTS, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, nargs="+", default=[4, 8, 12, 16])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jh", type=float, default=None)
    ap.add_argument("--backend", default="auto")
    ap.add_argument(
        "--noise-samples",
        type=int,
        default=0,
        help="PennyLane real-noise subset size (0 = skip)",
    )
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    base = CFG
    if args.quick:
        base = replace(
            base,
            n_components=6,
            n_steps=3,
            window_size=6,
            n_train=1500,
            n_bootstrap=200,
        )
        args.qubits = [q for q in args.qubits if q <= 6] or [4, 6]

    jh = args.jh
    if jh is None:
        p = os.path.join(RESULTS, "edge_of_chaos.json")
        jh = json.load(open(p))["best_jh"] if os.path.exists(p) else 1.0

    rows = []
    for nq in args.qubits:
        cfg = replace(base, n_qubits=nq, n_components=min(nq, base.n_components))
        data = prepare_data(cfg)
        per_seed = []
        for s in range(args.seeds):
            r = run_single(
                cfg,
                data,
                seed=cfg.seed_res + s,
                jh=jh,
                backend=args.backend,
                pl_noise_samples=args.noise_samples,
            )
            per_seed.append(r)
        auprc = M.summarize_seeds([r["qrc"] for r in per_seed], "auprc")
        eds = M.summarize_seeds([r["qrc"] for r in per_seed], "eds")
        row = {
            "n_qubits": nq,
            "readout_dim": cfg.readout_dim,
            "auprc_mean": auprc["mean"],
            "auprc_ci": auprc["ci"],
            "eds_mean": eds["mean"],
            "eds_ci": eds["ci"],
        }
        if args.noise_samples and "qrc_noisy" in per_seed[0]:
            nz = M.summarize_seeds(
                [r["qrc_noisy"] for r in per_seed if "qrc_noisy" in r], "auprc"
            )
            row["auprc_noisy_mean"] = nz["mean"]
            row["auprc_noisy_ci"] = nz["ci"]
        rows.append(row)
        msg = (
            f"  {nq:2d} qubits (dim {cfg.readout_dim}): "
            f"AUPRC={row['auprc_mean']:.4f}+/-{row['auprc_ci']:.4f}  "
            f"EDS={row['eds_mean']:.4f}+/-{row['eds_ci']:.4f}"
        )
        if "auprc_noisy_mean" in row:
            msg += f"  | noisy AUPRC={row['auprc_noisy_mean']:.4f}+/-{row['auprc_noisy_ci']:.4f}"
        print(msg)

    with open(os.path.join(RESULTS, "scaling.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("\nSaved results/scaling.json")


if __name__ == "__main__":
    main()
