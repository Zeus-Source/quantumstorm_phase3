"""
Tail-skill metrics + statistical machinery.

The Phase-2 appendix claimed noise was "strictly beneficial" from single-run
deltas as small as AUROC +0.0001. No reviewer will accept that. Everything here
supports confidence intervals (bootstrap) and a paired test so every headline
number can be reported as mean +/- 95% CI, and every "A beats B" claim carries a
p-value. Under-claiming with error bars beats over-claiming without them.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, \
    precision_recall_curve


# --------------------------------------------------------------------------
# Point metrics
# --------------------------------------------------------------------------
def optimal_threshold(y_true, y_prob):
    """F1-optimal threshold. Select this on VALIDATION, then freeze for test."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    return float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5


def eds(y_true, y_prob, threshold):
    """Extreme Dependency Score at a FIXED (val-selected) threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    p_hit = tp / (tp + fn + 1e-10)
    f_base = y_true.mean()
    if p_hit < 1e-10 or f_base < 1e-10:
        return -1.0
    return float(2 * np.log(p_hit + 1e-10) /
                 (np.log(f_base + 1e-10) + np.log(p_hit + 1e-10) + 1e-10) - 1)


def bss99(y_true, y_prob):
    mask = y_true == 1
    if mask.sum() == 0:
        return np.nan
    bs_model = np.mean((y_prob[mask] - 1) ** 2)
    bs_clim = np.mean((y_true.mean() - 1) ** 2)
    return float(1 - bs_model / (bs_clim + 1e-10))


def tw_crps(y_cont, y_prob, thr):
    n = min(len(y_cont), len(y_prob))
    y_cont, y_prob = y_cont[:n], y_prob[:n]
    w = np.maximum(0, y_cont - thr) / (thr + 1e-10)
    brier = (y_prob - (y_cont >= thr).astype(float)) ** 2
    return float(np.mean(w * brier))


def all_metrics(y_true, y_prob, threshold=None, y_cont=None, thr_cont=None):
    n = min(len(y_true), len(y_prob))
    y_true, y_prob = y_true[:n], y_prob[:n]
    if threshold is None:
        threshold = optimal_threshold(y_true, y_prob)
    out = {
        "auroc": roc_auc_score(y_true, y_prob) if y_true.sum() else 0.0,
        "auprc": average_precision_score(y_true, y_prob) if y_true.sum() else 0.0,
        "eds": eds(y_true, y_prob, threshold),
        "bss99": bss99(y_true, y_prob),
    }
    if y_cont is not None and thr_cont is not None:
        out["tw_crps"] = tw_crps(y_cont, y_prob, thr_cont)
    return out


# --------------------------------------------------------------------------
# Bootstrap confidence intervals + paired test
# --------------------------------------------------------------------------
def bootstrap_ci(y_true, y_prob, metric="auprc", n_boot=2000, seed=0,
                 threshold=None):
    """Percentile bootstrap 95% CI for a single metric."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    fn = {
        "auroc": lambda yt, yp: roc_auc_score(yt, yp) if yt.sum() else np.nan,
        "auprc": lambda yt, yp: average_precision_score(yt, yp) if yt.sum() else np.nan,
        "eds": lambda yt, yp: eds(yt, yp, threshold or optimal_threshold(yt, yp)),
        "bss99": bss99,
    }[metric]
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            vals.append(fn(y_true[idx], y_prob[idx]))
        except Exception:
            pass
    vals = np.array([v for v in vals if np.isfinite(v)])
    point = fn(y_true, y_prob)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi),
            "std": float(vals.std())}


def paired_bootstrap_test(y_true, prob_a, prob_b, metric="auprc",
                          n_boot=2000, seed=0):
    """One-sided paired bootstrap: P(metric_a <= metric_b) under resampling.
    Small p => model A significantly better than B on this metric."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    fn = {
        "auroc": lambda yt, yp: roc_auc_score(yt, yp) if yt.sum() else np.nan,
        "auprc": lambda yt, yp: average_precision_score(yt, yp) if yt.sum() else np.nan,
    }[metric]
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            diffs.append(fn(y_true[idx], prob_a[idx]) - fn(y_true[idx], prob_b[idx]))
        except Exception:
            pass
    diffs = np.array([d for d in diffs if np.isfinite(d)])
    point = fn(y_true, prob_a) - fn(y_true, prob_b)
    p = float(np.mean(diffs <= 0))  # frac of resamples where A not better
    return {"delta": float(point), "ci_lo": float(np.percentile(diffs, 2.5)),
            "ci_hi": float(np.percentile(diffs, 97.5)), "p_value": p}


def summarize_seeds(rows, metric):
    """mean +/- 95% CI across seeds (t-based) for a metric column."""
    v = np.array([r[metric] for r in rows if np.isfinite(r.get(metric, np.nan))])
    if len(v) == 0:
        return {"mean": np.nan, "ci": np.nan, "n": 0}
    mean = v.mean()
    ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return {"mean": float(mean), "ci": float(ci), "n": int(len(v)),
            "values": v.tolist()}
