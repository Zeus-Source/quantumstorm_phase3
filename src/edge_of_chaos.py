"""
Edge-of-many-body-chaos selection of J/h via the consecutive level-spacing
ratio <r>.  This is QuantumStorm's most original contribution: instead of a
blind grid search over J/h, pick the coupling that puts the reservoir spectrum
in the Wigner-Dyson crossover band <r> in [0.45, 0.50].

Poisson (integrable) ~ 0.386, GOE (chaotic) ~ 0.5307.

For the write-up you must show (a) the <r> vs J/h curve, and (b) that task
performance actually peaks in the selected band (ablation lives in
scripts/run_edge_of_chaos.py). A pretty selection curve that does NOT predict
performance is worth reporting honestly, not hiding.
"""
import numpy as np

try:
    import cupy as _cp
    _HAS_CUPY = True
except Exception:
    _cp = None
    _HAS_CUPY = False


def _build_H(J, h, n, adj, xp):
    dim = 2 ** n
    idx = xp.arange(dim, dtype=xp.int64)
    H = xp.zeros((dim, dim), dtype=xp.float64)
    # transverse field: off-diagonal single spin flips
    for i in range(n):
        flipped = idx ^ (1 << (n - i - 1))
        H[idx, flipped] += h
    # Ising ZZ: diagonal
    diag = xp.zeros(dim, dtype=xp.float64)
    for i in range(n):
        zi = (1 - 2 * ((idx >> (n - i - 1)) & 1)).astype(xp.float64)
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                zj = (1 - 2 * ((idx >> (n - j - 1)) & 1)).astype(xp.float64)
                diag += float(J[i, j]) * zi * zj
    H[idx, idx] += diag
    return H


def r_statistic(J, h, n, adj, backend="auto"):
    """Mean consecutive level-spacing ratio of the TFIM spectrum."""
    xp = _cp if (backend == "cupy" or (backend == "auto" and _HAS_CUPY)) else np
    H = _build_H(J, h, n, adj, xp)
    evals = xp.linalg.eigvalsh(H)
    evals = np.asarray(_cp.asnumpy(evals) if xp is _cp else evals)
    if xp is _cp:
        _cp.get_default_memory_pool().free_all_blocks()
    s = np.diff(np.sort(evals))
    s = s[s > 1e-10]
    if len(s) < 2:
        return 0.4
    r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
    return float(np.mean(r))


def select_jh(cfg, n_diag=12, ratios=None, target=(0.45, 0.50), backend="auto",
              verbose=True):
    """Sweep J/h, return the ratio whose <r> is closest to the band centre.

    We diagonalise at n_diag qubits (dim 2^12 = 4096 is cheap) because the
    Wigner-Dyson crossover ratio is size-robust for random-graph TFIM, then
    apply the chosen J/h to the full-size reservoir.
    """
    if ratios is None:
        ratios = np.linspace(0.2, 3.0, 20)
    rng = np.random.RandomState(cfg.seed_res + 1)
    adj = np.triu(rng.binomial(1, cfg.p_connect, (n_diag, n_diag)), k=1)
    J0 = rng.randn(n_diag, n_diag) * adj

    rs = []
    for ratio in ratios:
        r = r_statistic(J0 * ratio, cfg.h, n_diag, adj, backend)
        rs.append(r)
        if verbose:
            flag = "  <- band" if target[0] <= r <= target[1] else ""
            print(f"  J/h={ratio:5.2f}  <r>={r:.4f}{flag}")
    rs = np.array(rs)
    centre = 0.5 * (target[0] + target[1])
    best = int(np.argmin(np.abs(rs - centre)))
    in_band = [(float(ratios[i]), float(rs[i])) for i in range(len(rs))
               if target[0] <= rs[i] <= target[1]]
    return {
        "best_jh": float(ratios[best]),
        "best_r": float(rs[best]),
        "ratios": ratios.tolist(),
        "r_values": rs.tolist(),
        "in_band": in_band,
    }
