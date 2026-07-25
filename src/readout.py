"""
Classical readout: degree-2 polynomial expansion of the reservoir observables
followed by a focal-weighted ridge regression solved in closed form
(reweighted normal equations). Only this layer is trained; the reservoir is
fixed and gradient-free.
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def poly_features(R, degree=2):
    return PolynomialFeatures(degree=degree, include_bias=False).fit_transform(R)


def focal_weights(y, factor=30.0):
    w = np.ones(len(y), dtype=np.float64)
    w[y == 1] = factor
    return w


def weighted_ridge(X, y, weights, alpha=1.0):
    """Closed-form weighted ridge: (X'WX + aI)^-1 X'Wy."""
    Xw = X * weights[:, None]
    A = Xw.T @ X + alpha * np.eye(X.shape[1])
    b = Xw.T @ y
    return np.linalg.solve(A, b)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_readout(R_train, y_train, R_val, y_val, cfg, alphas=None):
    """Fit poly+ridge, selecting ridge alpha on the VALIDATION set (never test).
    Returns (weights, poly_transformer, best_alpha)."""
    if alphas is None:
        alphas = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    from sklearn.metrics import average_precision_score

    poly = PolynomialFeatures(degree=cfg.poly_degree, include_bias=False)
    Xtr = poly.fit_transform(R_train)
    Xva = poly.transform(R_val)
    w_focal = focal_weights(y_train, cfg.focal_factor)

    best_alpha, best_score, best_w = alphas[0], -1.0, None
    for a in alphas:
        w = weighted_ridge(Xtr, y_train.astype(np.float64), w_focal, a)
        p = sigmoid(Xva @ w)
        score = average_precision_score(y_val, p) if y_val.sum() > 0 else 0.0
        if score > best_score:
            best_score, best_alpha, best_w = score, a, w
    return best_w, poly, best_alpha


def predict(R, poly, w):
    return sigmoid(poly.transform(R) @ w)
