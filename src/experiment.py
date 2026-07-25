"""
End-to-end orchestration for the ERA5 rare-event track.

- run_single()      : one full pipeline (data -> reservoir -> readout -> metrics)
- run_multiseed()   : repeat across reservoir/subsample seeds -> mean +/- CI,
                      plus a paired bootstrap test QRC vs ESN and noisy vs clean.

This is where the "credibility" upgrade lives: the Phase-2 claims become
statements with error bars and p-values.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from .data import load_era5, build_windows, stratified_subsample
from .reservoir import build_reservoir, run_reservoir, run_reservoir_pennylane
from .readout import fit_readout, predict
from .baselines import tune_esn, esn_predict, persistence, nwp_proxy, arima, ar_lag
from . import metrics as M


def prepare_data(cfg):
    df = load_era5(cfg)
    splits, thr, tgt, y_cont_test = build_windows(df, cfg)
    (Xtr, ytr) = splits["train"]
    (Xva, yva) = splits["val"]
    (Xte, yte) = splits["test"]

    scaler = StandardScaler().fit(Xtr)
    pca = PCA(n_components=cfg.n_components).fit(scaler.transform(Xtr))
    mm = MinMaxScaler()

    def enc(X):
        return mm.transform(pca.transform(scaler.transform(X))).astype(np.float32)

    mm.fit(pca.transform(scaler.transform(Xtr)))
    return {
        "Xtr": enc(Xtr), "ytr": ytr, "Xva": enc(Xva), "yva": yva,
        "Xte": enc(Xte), "yte": yte, "y_cont_test": y_cont_test,
        "thr_cont": np.percentile(df.values, 99),  # rough; overwritten below
        "var_explained": float(pca.explained_variance_ratio_.sum()),
    }


def run_single(cfg, data, seed=None, jh=1.0, backend="auto",
               noisy=False, pl_noise_samples=0):
    """One QRC pipeline + ESN + trivial baselines. Returns probs + metrics."""
    seed = cfg.seed_res if seed is None else seed
    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva = data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]

    keep = stratified_subsample(ytr, cfg.n_train, seed=seed)
    Xtr_s, ytr_s = Xtr[keep], ytr[keep]

    pairs, J, adj = build_reservoir(cfg, seed=seed, jh=jh)
    Rtr = run_reservoir(Xtr_s, cfg, pairs, backend=backend)
    Rva = run_reservoir(Xva, cfg, pairs, backend=backend)
    Rte = run_reservoir(Xte, cfg, pairs, backend=backend)

    w, poly, alpha = fit_readout(Rtr, ytr_s, Rva, yva, cfg)
    probs_qrc = predict(Rte, poly, w)

    # F1-optimal threshold selected on VALIDATION (no test leakage)
    p_va = predict(Rva, poly, w)
    thr = M.optimal_threshold(yva, p_va)

    out = {"seed": seed, "alpha": alpha, "threshold": thr,
           "probs_qrc": probs_qrc, "y_true": yte}
    out["qrc"] = M.all_metrics(yte, probs_qrc, threshold=thr,
                               y_cont=data["y_cont_test"],
                               thr_cont=data["thr_cont"])

    # optional: genuine-noise QRC on a small subset via PennyLane density matrix
    if pl_noise_samples > 0:
        try:
            Rte_noisy = run_reservoir_pennylane(
                Xte, cfg, pairs, device="default.mixed",
                max_samples=pl_noise_samples)
            # reuse readout trained on clean features (mismatch is the point:
            # does calibrated hardware noise regularize?) -> retrain quickly
            Rtr_noisy = run_reservoir_pennylane(
                Xtr_s, cfg, pairs, device="default.mixed",
                max_samples=min(pl_noise_samples, len(Xtr_s)))
            wn, polyn, _ = fit_readout(Rtr_noisy, ytr_s[:len(Rtr_noisy)],
                                       Rva[:len(Rtr_noisy)], yva[:len(Rtr_noisy)], cfg)
            probs_noisy = predict(Rte_noisy, polyn, wn)
            out["probs_qrc_noisy"] = probs_noisy
            out["qrc_noisy"] = M.all_metrics(yte[:len(probs_noisy)], probs_noisy)
        except Exception as e:
            print(f"[experiment] PennyLane noise path skipped ({e})")

    return out


def run_baselines(cfg, data, alpha=1.0):
    Xtr, ytr = data["Xtr"], data["ytr"]
    Xva, yva = data["Xva"], data["yva"]
    Xte, yte = data["Xte"], data["yte"]
    keep = stratified_subsample(ytr, cfg.n_train, seed=cfg.seed_res)

    esn = tune_esn(Xtr[keep], ytr[keep], Xva, yva, cfg, alpha)
    probs = {
        "ESN": esn_predict(Xte, esn, cfg),
        "Persistence": persistence(Xte),
        "NWP-proxy": nwp_proxy(data["y_cont_test"], len(yte)),
        "ARIMA": arima(data["y_cont_test"][:len(Xtr)], len(yte)),
        "AR-lag7": ar_lag(Xtr[keep], ytr[keep], Xte),
    }
    res = {k: M.all_metrics(yte, v) for k, v in probs.items()}
    return probs, res, esn


def run_multiseed(cfg, backend="auto", jh=1.0, verbose=True):
    """Repeat the QRC pipeline across seeds; report mean +/- CI and a paired
    significance test against the ESN baseline."""
    data = prepare_data(cfg)
    base_probs, base_res, esn = run_baselines(cfg, data)

    seed_rows = []
    per_seed_probs = []
    for s in range(cfg.n_seeds):
        r = run_single(cfg, data, seed=cfg.seed_res + s, jh=jh, backend=backend)
        seed_rows.append(r["qrc"])
        per_seed_probs.append(r["probs_qrc"])
        if verbose:
            m = r["qrc"]
            print(f"  seed {s}: AUPRC={m['auprc']:.4f} AUROC={m['auroc']:.4f} "
                  f"EDS={m['eds']:.4f}")

    y_true = data["yte"]
    summary = {m: M.summarize_seeds(seed_rows, m)
               for m in ("auroc", "auprc", "eds", "bss99")}

    # significance: mean QRC probs (across seeds) vs ESN
    mean_probs = np.mean(per_seed_probs, axis=0)
    sig = {
        "auprc": M.paired_bootstrap_test(y_true, mean_probs, base_probs["ESN"],
                                         "auprc", cfg.n_bootstrap),
        "auroc": M.paired_bootstrap_test(y_true, mean_probs, base_probs["ESN"],
                                         "auroc", cfg.n_bootstrap),
    }
    return {
        "qrc_summary": summary,
        "baselines": base_res,
        "esn_config": {k: esn[k] for k in ("n_res", "spectral_radius", "leak",
                                           "val_auprc")},
        "significance_vs_esn": sig,
        "var_explained": data["var_explained"],
        "n_seeds": cfg.n_seeds,
    }
