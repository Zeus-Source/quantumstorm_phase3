#!/usr/bin/env python3
"""Main ERA5 rare-event benchmark: multi-seed QRC vs all classical baselines,
with confidence intervals and a paired significance test.

    python scripts/run_benchmark.py --seeds 5 --jh 0.20
    python scripts/run_benchmark.py --quick        # tiny config, ~1 min CPU

Reads J/h from results/edge_of_chaos.json if --jh is not given.
"""
import os, sys, json, argparse
from dataclasses import replace
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CFG
from src.experiment import run_multiseed

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)


def resolve_jh(arg):
    if arg is not None:
        return arg
    p = os.path.join(RESULTS, "edge_of_chaos.json")
    if os.path.exists(p):
        return json.load(open(p))["best_jh"]
    print("[warn] no edge_of_chaos.json; defaulting J/h=1.0. Run "
          "scripts/run_edge_of_chaos.py first.")
    return 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--jh", type=float, default=None)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = CFG
    if args.quick:
        cfg = replace(cfg, n_qubits=6, n_components=6, n_steps=3,
                      window_size=6, n_train=1500, n_seeds=2, n_bootstrap=200)
    if args.seeds:
        cfg = replace(cfg, n_seeds=args.seeds)
    jh = resolve_jh(args.jh)

    print(f"Config: {cfg.n_qubits} qubits | {cfg.n_seeds} seeds | J/h={jh} | "
          f"backend={args.backend}")
    out = run_multiseed(cfg, backend=args.backend, jh=jh)

    print("\n" + "=" * 62)
    print("QRC (mean +/- 95% CI across seeds)")
    print("=" * 62)
    for m in ("auroc", "auprc", "eds", "bss99"):
        s = out["qrc_summary"][m]
        print(f"  {m.upper():7s}: {s['mean']:.4f} +/- {s['ci']:.4f}  (n={s['n']})")

    print("\nBaselines (single run):")
    for name, r in out["baselines"].items():
        print(f"  {name:12s}: AUPRC={r['auprc']:.4f}  AUROC={r['auroc']:.4f}  "
              f"EDS={r['eds']:.4f}")

    print(f"\nTuned ESN: {out['esn_config']}")
    print("\nSignificance QRC vs ESN (paired bootstrap):")
    for m, s in out["significance_vs_esn"].items():
        verdict = "SIGNIFICANT" if s["p_value"] < 0.05 else "not significant"
        print(f"  {m.upper()}: delta={s['delta']:+.4f}  "
              f"95%CI[{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}]  "
              f"p={s['p_value']:.3f}  -> {verdict}")

    with open(os.path.join(RESULTS, "benchmark.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved results/benchmark.json")


if __name__ == "__main__":
    main()
