# QuantumStorm — Quantum Reservoir Computing for Rare-Event Weather Forecasting

**Team:** QuantumStorm (Justus Lau, Niloy Kumar Mondal)
**Challenge:** qBraid × MITRE × JonesTrading — Quantum Reservoir Computing for Time-Series Intelligence
**Track:** B — Weather / Atmospheric Forecasting
**Sub-problem:** Detection of 99th-percentile extreme-precipitation events from ERA5 fields (rare-event tail skill)

[![Launch on qBraid](https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png)](https://account.qbraid.com/?gitHubUrl=https://github.com/Zeus-Source/quantumstorm_phase3.git)

*Click the badge above to launch this complete environment, dependencies, and registered Python kernels instantly in qBraid Lab.*

---

## What this is

A fixed transverse-field Ising **quantum reservoir** (gradient-free) with a trained
linear/polynomial ridge **readout**, applied to rare-event extreme-precipitation
detection, benchmarked against strong classical baselines with **confidence
intervals and significance tests**. J/h is chosen by an **edge-of-many-body-chaos**
level-spacing criterion rather than a blind grid search.

## Setup & Quickstart (CPU, ~1–2 min, no data download needed)

You can set up the environment either via the standard `requirements.txt` or using the **recommended qBraid environment manager script** which creates a persistent, registered qBraid kernel:

### Recommended: Automatic qBraid Environment Setup
```bash
chmod +x setup_env.sh
./setup_env.sh
# Activate and use the environment's python:
/home/jovyan/.qbraid/environments/quantumstorm/bin/python scripts/run_edge_of_chaos.py --n-diag 8
/home/jovyan/.qbraid/environments/quantumstorm/bin/python scripts/run_benchmark.py --quick --backend numpy
/home/jovyan/.qbraid/environments/quantumstorm/bin/python scripts/run_mnist.py --qubits 4 6 --backend numpy
```

### Alternative: Standard manual setup
```bash
pip install -r requirements.txt
python scripts/run_edge_of_chaos.py --n-diag 8          # select J/h
python scripts/run_benchmark.py --quick --backend numpy # QRC vs baselines
python scripts/run_mnist.py --qubits 4 6 --backend numpy
```

**Data Loading & Zero-Configuration Reproducibility:** To make evaluation incredibly simple for the jury, the pre-processed, real ERA5 climate feature dataset is **already pre-computed and packaged inside this repository at `data_cache/era5_features.pkl`**. When you run any of the quickstart scripts above, they will automatically detect and load this real cached dataset, allowing you to reproduce our exact headline benchmark results instantly with zero configuration or API setup required. 

*(If this cached file is ever deleted, the data layer (`src/data.py`) will automatically fall back to generating a deterministic synthetic ERA5-like series on-the-fly, or download the raw meteorological fields directly from the Copernicus Climate Data Store if a standard `CDSAPI_KEY` is configured in your environment).*

## Full run on qBraid (GPU + real ERA5)

If you wish to completely delete our cached dataset and regenerate the pre-processed `data_cache/era5_features.pkl` from scratch by pulling raw NetCDF climate fields directly from the Copernicus Climate Data Store (CDS), follow these steps:

1. **Get Copernicus Access:** Register for a free account at [Copernicus CDS](https://cds.climate.copernicus.eu). Retrieve your Personal Access Token from your Profile page and accept the ERA5 dataset licenses.
2. **Configure API Key:** Export your CDS key as an environment variable or create a `.env` file (copied from `.env.example`):
   ```bash
   export CDSAPI_KEY="your-personal-access-token"
   ```
3. **Delete Existing Cache:**
   ```bash
   rm -f data_cache/era5_features.pkl
   ```
4. **Run the Data Download & Compilation Script:**
   ```bash
   python scripts/era5_download.py
   ```
   *This script downloads hourly atmospheric variables for North Atlantic, Bay of Bengal, and South Asia region slices across all active years (2011-2019), computes 850hPa wind speeds, performs temporal resampling to 12-hour intervals, matches coordinate systems, cleans any missing rows, and writes the completed consolidated pickle to `data_cache/era5_features.pkl`.*

5. **Reproduce All Benchmark Results:**
   Once regenerated, run the complete QRC pipeline reproducing sweeps, multiseed benchmarks, and standard MNIST validations:
   ```bash
   python skill/skill.py configure_reservoir --n-diag 12
   python skill/skill.py reproduce_benchmark --seeds 5
   python scripts/run_scaling.py --qubits 4 8 12 16 --seeds 3 --noise-samples 300
   python scripts/run_mnist.py --qubits 5 10 15
   ```
5. Hardware-validation row (optional): the PennyLane path in
   `src/reservoir.py::run_reservoir_pennylane` accepts a real device string
   (e.g. an IBM/IonQ backend via `pennylane-qiskit` / the qBraid provider). Run a
   small subset and record qubit count, depth, shots, and wall-clock.

## qBraid Skill (agent-executable)

`skill/skill.py` exposes four agent-callable entry points — see `skill/SKILL.md`:

| entry point | what it does |
|---|---|
| `configure_reservoir` | edge-of-chaos J/h selection |
| `run_sweep` | scaling across qubit counts (with CIs) |
| `evaluate_tail_metrics` | all tail metrics, QRC + baselines |
| `reproduce_benchmark` | full headline result + significance |

## Repository layout

```
config.py                 frozen protocol (split, qubits, seeds) — SINGLE source of truth
src/data.py               ERA5 loader (env key, cache, synthetic fallback) + windowing
src/reservoir.py          TFIM reservoir: numpy/cupy statevector + PennyLane noise/hardware
src/edge_of_chaos.py      <r> level-spacing J/h selection
src/readout.py            poly features + focal-weighted ridge (closed form)
src/baselines.py          ESN (tuned), Persistence, ARIMA, NWP-proxy, AR
src/metrics.py            tail metrics + bootstrap CIs + paired test
src/mnist_benchmark.py    REQUIRED common benchmark (same reservoir)
src/experiment.py         orchestration + multi-seed harness
scripts/                  CLI wrappers -> results/*.json
skill/                    qBraid Skill (4 entry points)
```

## Scaling Study & Benchmark Results

The following table summarizes the out-of-sample test precision (AUPRC) across different qubit counts ($N$) compared to classical baselines. The QRC (Quantum Reservoir Computing) model scales logarithmically with the number of qubits ($N$), eventually outperforming all classical baselines at $N = 16$.

| Model | $N{=}4$ | $N{=}6$ | $N{=}10$ | $N{=}16$ | Best classical |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **QRC (emulated)** | 0.2402 | 0.2946 | 0.3850 | **0.4561** | — |
| **ESN / AR-lag7** | 0.4101 / 0.4204 | 0.4101 / 0.4204 | 0.4101 / 0.4204 | 0.4101 / 0.4204 | 0.4204 |
| **ARIMA / NWP / Persist.** | ≤ 0.063 | ≤ 0.063 | ≤ 0.063 | ≤ 0.063 | — |

### LaTeX Table Code

```latex
\begin{table}[t]
\centering\small
\caption{Out-of-sample test AUPRC. QRC = exact GPU-emulated statevector; the fit is $a\ln N+b$ ($R^2\!\approx\!0.999$).}
\label{tab:scaling}
\begin{tabular}{lccccc}
\toprule
 & $N{=}4$ & $N{=}6$ & $N{=}10$ & $N{=}16$ & Best classical \\
\midrule
QRC (emulated) & 0.2402 & 0.2946 & 0.3850 & \textbf{0.4561} & --- \\
ESN / AR-lag7 & 0.4101 / 0.4204 & 0.4101 / 0.4204 & 0.4101 / 0.4204 & 0.4101 / 0.4204 & 0.4204 \\
ARIMA / NWP / Persist. & \le 0.063 & \le 0.063 & \le 0.063 & \le 0.063 & --- \\
\bottomrule
\end{tabular}
\end{table}
```

## Expected outputs

Each script writes JSON to `results/`: `edge_of_chaos.json`, `benchmark.json`
(QRC mean±CI, baselines, QRC-vs-ESN p-values), `scaling.json`, `mnist.json`,
plus `results/edge_of_chaos.png`.

## Known limitations / honesty notes

- **Data reproducibility:** real ERA5 needs a Copernicus account + download.
  Judges can reproduce the full pipeline on the synthetic fallback; the exact
  ERA5 numbers require the cached `data_cache/era5_features.pkl` (bundle it or
  supply a CDS key).
- **Noise model:** the fast statevector path is noiseless. Realistic depolarizing
  + amplitude-damping results come from the PennyLane density-matrix path on a
  subset (compute-bound), so noise-study sample sizes are smaller than the
  noiseless runs — CIs reflect this.
- **Scale:** exact statevector is limited to ≈16 qubits on a single GPU; larger
  reservoirs need tensor-network or hardware backends.
- The MNIST offline fallback is 8×8 digits; the submission run should fetch real
  `mnist_784` for cross-team comparability.
