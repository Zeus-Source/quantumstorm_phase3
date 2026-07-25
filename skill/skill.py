#!/usr/bin/env python3
"""
QuantumStorm qBraid Skill — agent-executable entry points.

The challenge asks for "a structured, agent-executable package that allows an AI
coding agent to navigate the codebase, configure the reservoir, run training, and
reproduce results end-to-end." These four functions are that interface. Each is
importable AND callable from the CLI:

    python skill/skill.py configure_reservoir --n-diag 12
    python skill/skill.py run_sweep --qubits 4 8 12 16 --seeds 3
    python skill/skill.py evaluate_tail_metrics
    python skill/skill.py reproduce_benchmark --seeds 5

NOTE: confirm the exact packaging metadata qBraid expects for a "Skill" against
the current docs at https://docs.qbraid.com before final upload; the logic below
is the functional core and is independent of that wrapper.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from config import CFG
from src.edge_of_chaos import select_jh
from src.experiment import run_multiseed, prepare_data, run_single, run_baselines
from src import metrics as M

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS, exist_ok=True)


def _save(name, obj):
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=2, default=str)


def configure_reservoir(n_diag=12, backend="auto", **_):
    """Select J/h via the edge-of-chaos criterion and persist the config."""
    sel = select_jh(CFG, n_diag=n_diag, backend=backend)
    cfg_out = {"n_qubits": CFG.n_qubits, "n_steps": CFG.n_steps,
               "best_jh": sel["best_jh"], "best_r": sel["best_r"]}
    _save("reservoir_config.json", cfg_out)
    print(json.dumps(cfg_out, indent=2))
    return cfg_out


def run_sweep(qubits=(4, 8, 12, 16), seeds=3, jh=None, backend="auto", **_):
    """Scaling sweep across qubit counts with per-seed CIs."""
    if jh is None:
        p = os.path.join(RESULTS, "reservoir_config.json")
        jh = json.load(open(p))["best_jh"] if os.path.exists(p) else 1.0
    rows = []
    for nq in qubits:
        cfg = replace(CFG, n_qubits=nq, n_components=min(nq, CFG.n_components))
        data = prepare_data(cfg)
        per = [run_single(cfg, data, seed=CFG.seed_res + s, jh=jh, backend=backend)
               for s in range(seeds)]
        a = M.summarize_seeds([r["qrc"] for r in per], "auprc")
        rows.append({"n_qubits": nq, "auprc_mean": a["mean"], "auprc_ci": a["ci"]})
        print(f"  {nq} qubits: AUPRC {a['mean']:.4f} +/- {a['ci']:.4f}")
    _save("sweep.json", rows)
    return rows


def evaluate_tail_metrics(jh=None, backend="auto", **_):
    """Compute all tail metrics for QRC + baselines on the frozen test split."""
    if jh is None:
        p = os.path.join(RESULTS, "reservoir_config.json")
        jh = json.load(open(p))["best_jh"] if os.path.exists(p) else 1.0
    data = prepare_data(CFG)
    probs, base_res, _ = run_baselines(CFG, data)
    r = run_single(CFG, data, seed=CFG.seed_res, jh=jh, backend=backend)
    out = {"QRC": r["qrc"], **base_res}
    _save("tail_metrics.json", out)
    for name, m in out.items():
        print(f"  {name:12s} AUPRC={m['auprc']:.4f} AUROC={m['auroc']:.4f} "
              f"EDS={m['eds']:.4f} BSS99={m['bss99']:.4f}")
    return out


def reproduce_benchmark(seeds=5, jh=None, backend="auto", **_):
    """End-to-end headline result: multi-seed QRC vs baselines + significance."""
    if jh is None:
        p = os.path.join(RESULTS, "reservoir_config.json")
        jh = json.load(open(p))["best_jh"] if os.path.exists(p) else 1.0
    cfg = replace(CFG, n_seeds=seeds)
    out = run_multiseed(cfg, backend=backend, jh=jh)
    _save("benchmark.json", out)
    s = out["qrc_summary"]["auprc"]
    sig = out["significance_vs_esn"]["auprc"]
    print(f"QRC AUPRC {s['mean']:.4f} +/- {s['ci']:.4f} | "
          f"ESN {out['baselines']['ESN']['auprc']:.4f} | "
          f"delta {sig['delta']:+.4f} (p={sig['p_value']:.3f})")
    return out


ENTRY = {"configure_reservoir": configure_reservoir, "run_sweep": run_sweep,
         "evaluate_tail_metrics": evaluate_tail_metrics,
         "reproduce_benchmark": reproduce_benchmark}


def main():
    ap = argparse.ArgumentParser(description="QuantumStorm qBraid Skill")
    ap.add_argument("entry", choices=list(ENTRY))
    ap.add_argument("--n-diag", type=int, default=12)
    ap.add_argument("--qubits", type=int, nargs="+", default=[4, 8, 12, 16])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jh", type=float, default=None)
    ap.add_argument("--backend", default="auto")
    a = ap.parse_args()
    kw = dict(n_diag=a.n_diag, qubits=tuple(a.qubits), seeds=a.seeds,
              jh=a.jh, backend=a.backend)
    ENTRY[a.entry](**kw)


if __name__ == "__main__":
    main()
