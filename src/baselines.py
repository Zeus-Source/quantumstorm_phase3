"""
Track-B classical baselines on the SAME problem instance as the QRC.

The one that decides whether you have a story is the ESN: it is the classical
twin of a quantum reservoir, so "we beat a properly-tuned ESN" is the entire
justification for QRC. An under-tuned ESN is the easiest way to fake an
advantage and the easiest thing for a reviewer to catch, so tune_esn() sweeps
spectral radius / leak / size on the VALIDATION set.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from .readout import weighted_ridge, focal_weights, sigmoid


# --------------------------------------------------------------------------
# Echo State Network (classical reservoir)
# --------------------------------------------------------------------------
def build_esn(n_reservoir, n_in, spectral_radius=0.9, seed=42):
    rng = np.random.RandomState(seed)
    W = rng.randn(n_reservoir, n_reservoir).astype(np.float32)
    W *= spectral_radius / (np.max(np.abs(np.linalg.eigvals(W))) + 1e-9)
    W_in = (rng.randn(n_reservoir, n_in) * 0.1).astype(np.float32)
    return W, W_in


def run_esn(X, W, W_in, window_size, leak=1.0):
    """Temporal ESN states over a sliding window (numpy)."""
    n, d = X.shape
    n_res = W.shape[0]
    pad = np.zeros((window_size - 1, d), dtype=np.float32)
    Xp = np.concatenate([pad, X.astype(np.float32)], axis=0)
    states = np.zeros((n, n_res), dtype=np.float32)
    for start in range(n):
        s = np.zeros(n_res, dtype=np.float32)
        for t in range(window_size):
            u = Xp[start + t]
            s = (1 - leak) * s + leak * np.tanh(W @ s + W_in @ u)
        states[start] = s
    return states


def tune_esn(X_tr, y_tr, X_va, y_va, cfg, alpha=1.0):
    """Grid-search ESN hyperparameters on validation AUPRC. Returns best config
    plus fitted readout weights."""
    from sklearn.metrics import average_precision_score
    n_in = X_tr.shape[1]
    w_focal = focal_weights(y_tr, cfg.focal_factor)
    best = None
    for n_res in (100, 200):
        for sr in (0.8, 0.95, 1.1):
            for leak in (0.5, 1.0):
                W, W_in = build_esn(n_res, n_in, sr, cfg.seed_res)
                Rtr = run_esn(X_tr, W, W_in, cfg.window_size, leak)
                Rva = run_esn(X_va, W, W_in, cfg.window_size, leak)
                w = weighted_ridge(Rtr, y_tr.astype(np.float64), w_focal, alpha)
                p = sigmoid(Rva @ w)
                score = average_precision_score(y_va, p) if y_va.sum() else 0.0
                if best is None or score > best["val_auprc"]:
                    best = dict(n_res=n_res, spectral_radius=sr, leak=leak,
                                val_auprc=float(score), W=W, W_in=W_in, w=w)
    return best


def esn_predict(X, esn, cfg):
    R = run_esn(X, esn["W"], esn["W_in"], cfg.window_size, esn["leak"])
    return sigmoid(R @ esn["w"])


# --------------------------------------------------------------------------
# Trivial / classical statistical baselines
# --------------------------------------------------------------------------
def persistence(X_test):
    thr = np.percentile(X_test[:, -1], 99)
    return (X_test[:, -1] >= thr).astype(np.float32)


def nwp_proxy(y_cont_train, n_test):
    from scipy.stats import norm
    mu, sd = float(y_cont_train.mean()), float(y_cont_train.std())
    thr = np.percentile(y_cont_train, 99)
    p = float(1.0 - norm.cdf(thr, mu, sd + 1e-8))
    return np.full(n_test, p, dtype=np.float32)


def arima(y_cont_train, n_test, order=(2, 0, 2), fallback=None):
    try:
        from statsmodels.tsa.arima.model import ARIMA
        fit = ARIMA(y_cont_train, order=order).fit()
        f = fit.forecast(steps=n_test).astype(np.float32)
        return (f - f.min()) / (f.max() - f.min() + 1e-8)
    except Exception as e:
        print(f"[baselines] ARIMA failed ({e}); using fallback")
        return fallback if fallback is not None else np.zeros(n_test, np.float32)


def ar_lag(X_tr, y_tr, X_te, lag=7):
    m = LinearRegression().fit(X_tr[:, -lag:], y_tr)
    return np.clip(m.predict(X_te[:, -lag:]), 0, 1)
