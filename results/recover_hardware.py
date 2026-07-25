import os
import json
import numpy as np
from qbraid_core.services.runtime import QuantumRuntimeClient

# Target results and directories
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
HARDWARE_4_PATH = os.path.join(RESULTS_DIR, "hardware_4qubits.json")
HARDWARE_10_PATH = os.path.join(RESULTS_DIR, "hardware_10qubits.json")
HARDWARE_16_PATH = os.path.join(RESULTS_DIR, "hardware_16qubits.json")

# 1. Map all completed Job IDs from your qBraid history
JOB_IDS_4 = [
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a59f823cd7f82e68c23dfbe",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a59f822cd7f82e68c23dfbb",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a59f822cd7f82e68c23dfb8",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a59f821cd7f82e68c23dfb5",
]

JOB_IDS_10 = [
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a20bfcd7f82e68c23e117",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a20becd7f82e68c23e114",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a20bccd7f82e68c23e111",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a20bccd7f82e68c23e10e",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a20bbcd7f82e68c23e10b",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a20bbcd7f82e68c23e108",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a20bacd7f82e68c23e105",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a20b9cd7f82e68c23e102",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a20b9cd7f82e68c23e0ff",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a20b8cd7f82e68c23e0fc",
]

JOB_IDS_16 = [
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a023ccd7f82e68c23e00f",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a023acd7f82e68c23e00c",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a0239cd7f82e68c23e009",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a0237cd7f82e68c23e006",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a0237cd7f82e68c23e003",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a0236cd7f82e68c23e000",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a0236cd7f82e68c23dffd",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a0235cd7f82e68c23dffa",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a0235cd7f82e68c23dff7",
    "aws:rigetti:qpu:cepheus-1-108q-dc31-qjob-6a5a0234cd7f82e68c23dff4",
]


def pull_counts_for_jobs(client, job_ids, n_qubits):
    out = np.zeros(
        (len(job_ids), n_qubits + n_qubits * (n_qubits - 1) // 2), dtype=np.float32
    )
    for k, job_id in enumerate(job_ids):
        res_data = client.get_job_result(job_id)
        counts = res_data.resultData.get(
            "measurementCounts"
        ) or res_data.resultData.get("counts")
        total_counts = sum(counts.values())

        # Z Expectation
        z_vals = []
        for i in range(n_qubits):
            bit_idx = n_qubits - 1 - i
            count_0 = sum(val for key, val in counts.items() if key[bit_idx] == "0")
            count_1 = total_counts - count_0
            z_vals.append((count_0 - count_1) / total_counts)

        # ZZ Expectation
        zz_vals = []
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                bit_i = n_qubits - 1 - i
                bit_j = n_qubits - 1 - j
                count_even = sum(
                    val for key, val in counts.items() if key[bit_i] == key[bit_j]
                )
                count_odd = total_counts - count_even
                zz_vals.append((count_even - count_odd) / total_counts)

        out[k] = np.array(z_vals + zz_vals, dtype=np.float32)
    return out


def process_qubit_run(client, job_ids, n_qubits, save_path):
    print(f"\nProcessing {n_qubits}-Qubit physical run using {len(job_ids)} jobs...")
    out = pull_counts_for_jobs(client, job_ids, n_qubits)

    import sys

    sys.path.insert(0, os.path.dirname(RESULTS_DIR))
    from config import CFG
    from dataclasses import replace
    from src.experiment import prepare_data
    from src.reservoir import build_reservoir, run_reservoir
    from src.readout import fit_readout, predict
    import src.metrics as M

    # Target configurations
    p_connect = 0.3 if n_qubits in [4, 10] else 0.5
    cfg = replace(
        CFG, n_qubits=n_qubits, n_steps=2, p_connect=p_connect, n_components=n_qubits
    )
    data = prepare_data(cfg)

    from scripts.run_hardware import stratified_subset

    sub = stratified_subset(data["yte"], len(job_ids), seed=cfg.seed_res)
    ysub = data["yte"][sub]
    Xsub = data["Xte"][sub]

    # Recreate Reservoir graph and exact state vector
    _, J0, adj = build_reservoir(cfg, seed=cfg.seed_res, jh=1.0)

    # Best J/h from runs
    jh_val = 2.0 if n_qubits == 4 else (0.8 if n_qubits == 10 else 0.2)
    pairs = [
        (i, j, float(J0[i, j] * jh_val))
        for i in range(cfg.n_qubits)
        for j in range(i + 1, cfg.n_qubits)
        if abs(J0[i, j]) > 1e-8
    ]

    R_sim = run_reservoir(Xsub, cfg, pairs, backend="auto")

    # Fit linear readout classically
    from src.data import stratified_subsample

    keep = stratified_subsample(data["ytr"], 4000, seed=cfg.seed_res)
    Rtr = run_reservoir(data["Xtr"][keep], cfg, pairs, backend="auto")
    Rva = run_reservoir(data["Xva"], cfg, pairs, backend="auto")
    w, poly, alpha = fit_readout(Rtr, data["ytr"][keep], Rva, data["yva"], cfg)
    thr = M.optimal_threshold(data["yva"], predict(Rva, poly, w))

    p_sim = predict(R_sim, poly, w)
    p_hw = predict(out, poly, w)

    m_sim = M.all_metrics(ysub, p_sim, threshold=thr)
    m_hw = M.all_metrics(ysub, p_hw, threshold=thr)

    corr = float(np.corrcoef(out.ravel(), R_sim.ravel())[0, 1])
    mae = float(np.mean(np.abs(out - R_sim)))

    print(f"  Fidelity HW vs Sim: Correlation = {corr:.4f}  |  MAE = {mae:.4f}")
    print(f"  AUPRC: Sim = {m_sim['auprc']:.4f}  |  Hardware QPU = {m_hw['auprc']:.4f}")

    results = {
        "backend": "aws:rigetti:qpu:cepheus-1-108q",
        "n_qubits": n_qubits,
        "n_steps": 2,
        "p_connect": p_connect,
        "best_jh": jh_val,
        "best_r": 0.4888 if n_qubits == 4 else (0.4154 if n_qubits == 10 else 0.4763),
        "shots": 4096,
        "n_test": len(job_ids),
        "n_extremes": int(ysub.sum()),
        "readout_corr_hw_vs_sim": corr,
        "readout_mae_hw_vs_sim": mae,
        "metrics_sim": m_sim,
        "metrics_hw": m_hw,
    }

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved JSON to: {save_path}")


def main():
    print("Initializing qBraid Core Runtime Client...")
    client = QuantumRuntimeClient()

    # Process all 3 physical QPU runs to make individual JSON validation structures
    process_qubit_run(client, JOB_IDS_4, 4, HARDWARE_4_PATH)
    process_qubit_run(client, JOB_IDS_10, 10, HARDWARE_10_PATH)
    process_qubit_run(client, JOB_IDS_16, 16, HARDWARE_16_PATH)

    print("\nAll 3 physical validation stages recovered successfully!")


if __name__ == "__main__":
    main()
