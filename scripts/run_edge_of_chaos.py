#!/usr/bin/env python3
"""Select J/h via the level-spacing ratio and save the curve.

    python scripts/run_edge_of_chaos.py --n-diag 12
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from src.edge_of_chaos import select_jh

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-diag", type=int, default=12)
    ap.add_argument("--points", type=int, default=20)
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()

    ratios = np.linspace(0.2, 3.0, args.points)
    sel = select_jh(CFG, n_diag=args.n_diag, ratios=ratios, backend=args.backend)
    print(f"\nSelected J/h = {sel['best_jh']:.3f}  (<r> = {sel['best_r']:.4f})")
    print(f"Ratios in Wigner-Dyson band [0.45,0.50]: {len(sel['in_band'])}")

    with open(os.path.join(RESULTS, "edge_of_chaos.json"), "w") as f:
        json.dump(sel, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 4))
        plt.plot(sel["ratios"], sel["r_values"], "o-", color="purple", lw=2)
        plt.axhspan(0.45, 0.50, alpha=0.2, color="green", label="target band")
        plt.axhline(0.386, ls="--", color="blue", label="Poisson")
        plt.axhline(0.5307, ls="--", color="red", label="GOE")
        plt.axvline(sel["best_jh"], ls=":", color="orange", lw=2,
                    label=f"selected J/h={sel['best_jh']:.2f}")
        plt.xlabel("J/h"); plt.ylabel(r"$\langle r \rangle$")
        plt.title(f"Edge-of-chaos selection (N={args.n_diag} exact diag)")
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(os.path.join(RESULTS, "edge_of_chaos.png"), dpi=150)
        print(f"Saved results/edge_of_chaos.png")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
