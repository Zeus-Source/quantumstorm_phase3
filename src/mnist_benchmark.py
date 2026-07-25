"""
MNIST common benchmark — REQUIRED of every team.

From the Aqora challenge page: "all participants implement a common MNIST digit
classification benchmark using their QRC architecture, providing a standardized
comparison across teams and validating that the quantum reservoir exhibits
sufficient expressivity." This was MISSING from the Phase-2 submission and is a
disqualification / big-score risk, so it is a first-class deliverable here.

We reuse the SAME TFIM reservoir: PCA/downsample each image to n_qubits
features, angle-encode, take <Z_i>,<Z_iZ_j> readout, then a multiclass linear
classifier. We report accuracy across qubit counts (expressivity vs. Hilbert
dimension) so it plugs straight into the scaling narrative.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

from .reservoir import build_reservoir, run_reservoir
from .readout import poly_features


def load_mnist(n_train=2000, n_test=1000, seed=0):
    """Real MNIST via OpenML if available; else sklearn digits (8x8) offline."""
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X, y = mnist.data.astype(np.float32), mnist.target.astype(int)
        source = "mnist_784"
    except Exception as e:
        print(f"[mnist] OpenML unavailable ({e}); falling back to sklearn digits (8x8)")
        from sklearn.datasets import load_digits
        d = load_digits()
        X, y = d.data.astype(np.float32), d.target.astype(int)
        source = "sklearn_digits"
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:n_train + n_test], y[n_train:n_train + n_test]
    return (Xtr, ytr), (Xte, yte), source


def run_mnist_qrc(cfg, n_qubits=10, n_train=2000, n_test=1000,
                  backend="auto", jh=1.0, seed=0):
    """Full QRC MNIST pipeline at a given qubit count. Returns accuracy + meta."""
    (Xtr, ytr), (Xte, yte), source = load_mnist(n_train, n_test, seed)

    # image -> n_qubits features in [0, 1]
    pca = PCA(n_components=n_qubits, random_state=seed).fit(Xtr)
    scaler = MinMaxScaler()
    Ftr = scaler.fit_transform(pca.transform(Xtr)).astype(np.float32)
    Fte = scaler.transform(pca.transform(Xte)).astype(np.float32)

    # reservoir at this qubit count
    cfg_q = _clone_with_qubits(cfg, n_qubits)
    pairs, _, _ = build_reservoir(cfg_q, seed=cfg.seed_res, jh=jh)
    Rtr = run_reservoir(Ftr, cfg_q, pairs, backend=backend)
    Rte = run_reservoir(Fte, cfg_q, pairs, backend=backend)

    Ptr = poly_features(Rtr, cfg.poly_degree)
    Pte = poly_features(Rte, cfg.poly_degree)

    clf = RidgeClassifier(alpha=1.0).fit(Ptr, ytr)
    acc = accuracy_score(yte, clf.predict(Pte))
    return {"n_qubits": n_qubits, "accuracy": float(acc),
            "readout_dim": cfg_q.readout_dim, "source": source,
            "n_train": len(ytr), "n_test": len(yte)}


def _clone_with_qubits(cfg, n_qubits):
    from dataclasses import replace
    return replace(cfg, n_qubits=n_qubits)


def mnist_scaling(cfg, qubit_counts=(4, 6, 8, 10), **kw):
    rows = []
    for nq in qubit_counts:
        r = run_mnist_qrc(cfg, n_qubits=nq, **kw)
        print(f"  MNIST {nq:2d} qubits: acc={r['accuracy']:.4f} "
              f"(readout {r['readout_dim']}, {r['source']})")
        rows.append(r)
    return rows
