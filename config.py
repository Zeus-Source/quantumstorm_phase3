"""
QuantumStorm Phase 3 — Central configuration (single source of truth).

Every script imports CFG from here. This is the fix for the Phase-2 problem
of three different train/val/test splits appearing in the same document:
the split is now defined exactly once, and the write-up must quote these
numbers verbatim.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Tuple
import json


@dataclass
class Config:
    # ---- Reservoir ----
    n_qubits: int = 16
    n_steps: int = 10  # Trotter steps
    dt: float = 0.1  # Trotter step size (natural units)
    h: float = 1.0  # transverse field
    p_connect: float = 0.5  # Erdos-Renyi edge probability
    seed_res: int = 42  # base reservoir seed

    # ---- Encoding / readout ----
    n_components: int = 16  # PCA components fed into the reservoir
    window_size: int = 28  # temporal multiplexing window (28 * 12h = 14 days)
    poly_degree: int = 2  # classical readout polynomial expansion
    focal_factor: float = 30.0  # rare-event up-weighting in the ridge normal equations

    # ---- Data / task ----
    target_var: str = "wind_speed850"  # target field prefix (extreme storms indicator)
    threshold_pct: float = 99.0  # extreme = >= 99th percentile of TRAIN target
    horizon: int = 5  # predict an extreme within the next `horizon` steps
    n_train: int = 20000  # cap on training samples after stratified subsampling

    # ---- FROZEN temporal split (quote these EXACT years in the write-up) ----
    years_train: Tuple[int, int] = (2011, 2015)  # inclusive
    years_val: Tuple[int, int] = (2016, 2017)
    years_test: Tuple[int, int] = (2018, 2019)

    # ---- Noise (realistic channels, used on the PennyLane density-matrix path) ----
    p_depol: float = (
        0.0091  # calibrated depolarizing probability (matched to ERA5 sigma)
    )
    gamma_ad: float = 0.005  # amplitude-damping rate per layer

    # ---- Statistics ----
    n_seeds: int = 5  # multi-seed repetitions for confidence intervals
    n_bootstrap: int = 2000  # bootstrap resamples for CIs and paired tests

    # ---- Reproducibility ----
    regions: dict = field(
        default_factory=lambda: {
            "north_atlantic": (5, 35, -90, -10),
            "bay_of_bengal": (5, 25, 80, 100),
            "south_asia": (10, 30, 70, 110),
        }
    )

    @property
    def readout_dim(self) -> int:
        n = self.n_qubits
        return n + n * (n - 1) // 2  # <Z_i> + <Z_i Z_j>

    def years(self, split: str) -> List[int]:
        lo, hi = getattr(self, f"years_{split}")
        return list(range(lo, hi + 1))

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2, default=str)


CFG = Config()

if __name__ == "__main__":
    print(CFG.to_json())
    print("readout_dim:", CFG.readout_dim)
    for s in ("train", "val", "test"):
        print(s, CFG.years(s))
