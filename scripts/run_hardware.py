#!/usr/bin/env python3
"""
Run the QuantumStorm edge-of-chaos QRC on real quantum hardware.

This implements the actual theoretical idea end-to-end on a QPU:
  1. Build the random-graph (chaotic) TFIM reservoir at the chosen qubit count.
  2. Select J/h by the level-spacing ratio <r> ON THAT EXACT GRAPH, so the
     reservoir sits at the edge of many-body chaos (the paper's core claim).
  3. Print the transpiled 2-qubit-gate / depth budget BEFORE running (feasibility
     gate). Real QPUs decohere past a few hundred 2-qubit gates.
  4. Train the linear readout on the FAST simulator over the full training subset
     (free, classical) -- only the readout is trained.
  5. Run a small STRATIFIED test subset on the backend (ideal Aer / device-noise
     Aer / real QPU), apply the trained readout, and report:
       - downstream tail metrics on hardware vs exact simulator,
       - readout fidelity (correlation / MAE) hardware vs simulator  <- the key
         hardware-validation signal, needs no labels,
       - resource numbers: qubits, ISA 2q gates, ISA depth, shots, wall-clock.

Examples
--------
    # free dry run: just the depth/feasibility estimate
    python scripts/run_hardware.py --backend aer_noisy:FakeBrisbane --qubits 12 \
        --steps 4 --p-connect 0.3 --dry-run

    # device-noise simulation (free, no QPU time), full metrics
    python scripts/run_hardware.py --backend aer_noisy:FakeBrisbane --qubits 12 \
        --steps 4 --p-connect 0.3 --n-test 120

    # REAL hardware (needs QISKIT_IBM_TOKEN or --token); one least-busy QPU
    python scripts/run_hardware.py --backend least_busy --qubits 12 --steps 3 \
        --p-connect 0.3 --n-test 80 --shots 4096
"""

import os, sys, json, time, argparse
from dataclasses import replace
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CFG
from src.experiment import prepare_data
from src.reservoir import build_reservoir, run_reservoir
from src.edge_of_chaos import r_statistic
from src.readout import fit_readout, predict
from src import metrics as M
import src.hardware_qiskit as HW

RESULTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RESULTS, exist_ok=True)


def select_jh_on_graph(J0, adj, cfg, ratios=None, target=0.475, verbose=True):
    """Edge-of-chaos J/h selection on the ACTUAL reservoir graph (exact diag,
    feasible up to ~13 qubits). Ties the theory directly to what runs on the QPU."""
    if ratios is None:
        ratios = np.linspace(0.2, 3.0, 15)
    rs = []
    for ratio in ratios:
        r = r_statistic(J0 * ratio, cfg.h, cfg.n_qubits, adj, backend="numpy")
        rs.append(r)
        if verbose:
            flag = "  <- edge of chaos" if 0.45 <= r <= 0.50 else ""
            print(f"    J/h={ratio:5.2f}  <r>={r:.4f}{flag}")
    rs = np.array(rs)
    best = int(np.argmin(np.abs(rs - target)))
    return float(ratios[best]), float(rs[best])


def stratified_subset(y, n, seed=0):
    """Test subset that actually contains extremes (they are ~1% of data, so a
    random 100-sample slice would have almost none and make AUPRC meaningless)."""
    rng = np.random.RandomState(seed)
    ext = np.where(y == 1)[0]
    norm = np.where(y == 0)[0]
    n_ext = min(len(ext), max(n // 3, 1))
    n_norm = min(len(norm), n - n_ext)
    idx = np.concatenate(
        [rng.choice(ext, n_ext, replace=False), rng.choice(norm, n_norm, replace=False)]
    )
    return np.sort(idx)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--backend",
        default="aer_noisy:FakeBrisbane",
        help="aer | aer_noisy:FakeName | least_busy | <ibm_backend>",
    )
    ap.add_argument("--qubits", type=int, default=12)
    ap.add_argument(
        "--steps", type=int, default=4, help="Trotter steps (lower = shallower)"
    )
    ap.add_argument(
        "--p-connect",
        type=float,
        default=0.3,
        help="graph edge prob (lower = fewer 2q gates, still chaotic)",
    )
    ap.add_argument("--n-test", type=int, default=120, help="hardware test subset size")
    ap.add_argument("--n-train", type=int, default=4000, help="readout training subset")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--opt-level", type=int, default=1)
    ap.add_argument("--jh", type=float, default=None, help="override edge-of-chaos J/h")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--token", default=None)
    ap.add_argument("--instance", default=None)
    ap.add_argument(
        "--force", action="store_true", help="run even if 2q budget is large"
    )
    ap.add_argument("--dry-run", action="store_true", help="only print feasibility")
    args = ap.parse_args()

    seed = CFG.seed_res if args.seed is None else args.seed
    cfg = replace(
        CFG,
        n_qubits=args.qubits,
        n_steps=args.steps,
        p_connect=args.p_connect,
        n_components=min(args.qubits, CFG.n_components),
    )

    # ---- 1-2. reservoir + edge-of-chaos J/h on the actual graph ----
    _, J0, adj = build_reservoir(cfg, seed=seed, jh=1.0)  # base couplings
    print(
        f"Reservoir: {cfg.n_qubits} qubits | random graph p={cfg.p_connect} | "
        f"{int((adj > 0).sum())} edges | {cfg.n_steps} Trotter steps"
    )
    if args.jh is not None:
        best_jh, best_r = args.jh, float("nan")
        print(f"Using provided J/h = {best_jh}")
    elif cfg.n_qubits <= 13:
        print("Edge-of-chaos selection on this graph:")
        best_jh, best_r = select_jh_on_graph(J0, adj, cfg)
    else:
        from src.edge_of_chaos import select_jh

        sel = select_jh(cfg, n_diag=12, backend="numpy", verbose=False)
        best_jh, best_r = sel["best_jh"], sel["best_r"]
        print(f"(>13 qubits: J/h extrapolated from N=12 diag)")
    print(f"Selected J/h = {best_jh:.3f}  (<r> = {best_r:.4f})\n")

    pairs = [
        (i, j, float(J0[i, j] * best_jh))
        for i in range(cfg.n_qubits)
        for j in range(i + 1, cfg.n_qubits)
        if abs(J0[i, j]) > 1e-8
    ]

    # ---- 3. feasibility gate ----
    backend = HW.get_backend(
        args.backend,
        service=(
            HW.make_service(args.token, args.instance)
            if args.backend in ("least_busy",) or args.backend.startswith("ibm")
            else None
        ),
        min_qubits=cfg.n_qubits,
    )
    cost = HW.transpile_cost(cfg, pairs, backend, cfg.n_qubits, args.opt_level)
    print("Transpiled cost on this backend:")
    for k in ("isa_depth", "isa_two_qubit_gates", "isa_two_qubit_depth"):
        print(f"    {k}: {cost[k]}")
    heavy = cost["isa_two_qubit_gates"] > 500
    if heavy:
        print(
            "  WARNING: >500 two-qubit gates -> expect heavy decoherence on real "
            "hardware. Reduce --steps / --qubits / --p-connect, or pass --force."
        )
    if args.dry_run:
        print("\n[dry-run] stopping before execution.")
        json.dump(
            {"cost": cost, "best_jh": best_jh, "best_r": best_r},
            open(os.path.join(RESULTS, "hardware_dryrun.json"), "w"),
            indent=2,
        )
        return
    if heavy and args.backend in ("least_busy",) and not args.force:
        print(
            "Refusing to spend QPU time on a likely-noise circuit. Use --force to override."
        )
        return

    # ---- 4. train readout on the fast simulator (full training subset) ----
    data = prepare_data(cfg)
    from src.data import stratified_subsample

    keep = stratified_subsample(data["ytr"], args.n_train, seed=seed)
    print(f"\nTraining readout on {len(keep)} samples (fast statevector sim)...")
    Rtr = run_reservoir(data["Xtr"][keep], cfg, pairs, backend="auto")
    Rva = run_reservoir(data["Xva"], cfg, pairs, backend="auto")
    w, poly, alpha = fit_readout(Rtr, data["ytr"][keep], Rva, data["yva"], cfg)
    thr = M.optimal_threshold(data["yva"], predict(Rva, poly, w))

    # ---- 5. run the test subset on the backend ----
    sub = stratified_subset(data["yte"], args.n_test, seed=seed)
    Xsub, ysub = data["Xte"][sub], data["yte"][sub]
    print(f"Test subset: {len(sub)} samples ({int(ysub.sum())} extremes)")

    R_sim = run_reservoir(Xsub, cfg, pairs, backend="auto")  # exact
    print(f"Running {len(sub)} circuits on '{args.backend}' ...")
    t0 = time.time()
    R_hw = HW.run_reservoir_qiskit(
        Xsub,
        cfg,
        pairs,
        backend,
        cfg.n_qubits,
        shots=args.shots,
        optimization_level=args.opt_level,
    )
    wall = time.time() - t0

    p_sim = predict(R_sim, poly, w)
    p_hw = predict(R_hw, poly, w)
    m_sim = M.all_metrics(ysub, p_sim, threshold=thr)
    m_hw = M.all_metrics(ysub, p_hw, threshold=thr)

    # ---- NEW: Classical Baseline Comparisons ----
    # Let's run the classical baselines on the same test subset
    print("\nRunning classical baselines on the same test subset...")
    from src.experiment import run_baselines

    # run_baselines expects full data, so we obtain its results and then slice for the sub indices
    base_probs, base_res, esn = run_baselines(cfg, data)

    # Evaluate classical baselines specifically on the identical test subset indices
    m_esn = M.all_metrics(
        ysub, base_probs["ESN"][sub], threshold=None
    )  # threshold=None defaults to optimal or standard
    m_persistence = M.all_metrics(ysub, base_probs["Persistence"][sub])
    m_nwp = M.all_metrics(ysub, base_probs["NWP-proxy"][sub])
    m_arima = M.all_metrics(ysub, base_probs["ARIMA"][sub])
    m_ar_lag = M.all_metrics(ysub, base_probs["AR-lag7"][sub])

    # readout fidelity (label-free hardware-validation signal)
    corr = float(np.corrcoef(R_hw.ravel(), R_sim.ravel())[0, 1])
    mae = float(np.mean(np.abs(R_hw - R_sim)))

    out = {
        "backend": args.backend,
        "n_qubits": cfg.n_qubits,
        "n_steps": cfg.n_steps,
        "p_connect": cfg.p_connect,
        "best_jh": best_jh,
        "best_r": best_r,
        "shots": args.shots,
        "n_test": int(len(sub)),
        "n_extremes": int(ysub.sum()),
        "isa_two_qubit_gates": cost["isa_two_qubit_gates"],
        "isa_depth": cost["isa_depth"],
        "wall_clock_s": round(wall, 2),
        "readout_corr_hw_vs_sim": corr,
        "readout_mae_hw_vs_sim": mae,
        "metrics_sim": m_sim,
        "metrics_hw": m_hw,
        "metrics_esn": m_esn,
        "metrics_persistence": m_persistence,
        "metrics_arima": m_arima,
    }
    print("\n" + "=" * 60)
    print(f"HARDWARE & CLASSICAL BASELINES COMPARISON  ({args.backend})")
    print("=" * 60)
    print(f"  readout fidelity vs exact sim: corr={corr:.4f}  MAE={mae:.4f}")
    print("-" * 60)
    print(f"  {'Model':<18} | {'AUPRC':<8} | {'AUROC':<8} | {'EDS':<8}")
    print("-" * 60)
    print(
        f"  {'QRC (Exact Sim)':<18} | {m_sim['auprc']:.4f} | {m_sim['auroc']:.4f} | {m_sim['eds']:.4f}"
    )
    print(
        f"  {'QRC (Hardware)':<18} | {m_hw['auprc']:.4f} | {m_hw['auroc']:.4f} | {m_hw['eds']:.4f}"
    )
    print(
        f"  {'ESN (Classical)':<18} | {m_esn['auprc']:.4f} | {m_esn['auroc']:.4f} | {m_esn['eds']:.4f}"
    )
    print(
        f"  {'ARIMA':<18} | {m_arima['auprc']:.4f} | {m_arima['auroc']:.4f} | {m_arima['eds']:.4f}"
    )
    print(
        f"  {'AR-lag7':<18} | {m_ar_lag['auprc']:.4f} | {m_ar_lag['auroc']:.4f} | {m_ar_lag['eds']:.4f}"
    )
    print(
        f"  {'Persistence':<18} | {m_persistence['auprc']:.4f} | {m_persistence['auroc']:.4f} | {m_persistence['eds']:.4f}"
    )
    print(
        f"  {'NWP-proxy':<18} | {m_nwp['auprc']:.4f} | {m_nwp['auroc']:.4f} | {m_nwp['eds']:.4f}"
    )
    print("-" * 60)
    print(
        f"  qubits={cfg.n_qubits}  2q-gates={cost['isa_two_qubit_gates']}  "
        f"depth={cost['isa_depth']}  shots={args.shots}  wall={wall:.1f}s"
    )
    json.dump(
        out, open(os.path.join(RESULTS, "hardware.json"), "w"), indent=2, default=str
    )
    print("\nSaved results/hardware.json")


if __name__ == "__main__":
    main()
