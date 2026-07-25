# src/hardware_qiskit.py
"""
Qiskit and qBraid compatibility layer for run_hardware.py.
Handles quantum device transpilation costs and circuit execution natively using qBraid runtime.
"""

import time
import numpy as np
from qiskit import QuantumCircuit
from qbraid.runtime import QbraidProvider


def make_service(token=None, instance=None):
    """
    Returns a QbraidProvider since qBraid manages credentials and submissions securely.
    """
    return QbraidProvider()


def get_backend(backend_name, service=None, min_qubits=1):
    """
    Returns a qBraid device wrapping the Qiskit or external backend.
    """
    provider = service or QbraidProvider()

    # Map friendly names / categories if needed
    if backend_name == "least_busy":
        backend_name = "ibm:ibm:qpu:fez"
    elif backend_name == "aer":
        backend_name = "qbraid:qbraid:sim:qir-sv"
    elif backend_name.startswith("aer_noisy"):
        backend_name = "qbraid:qbraid:sim:qir-sv"

    return provider.get_device(backend_name)


def transpile_cost(cfg, pairs, backend, n_qubits, opt_level=1):
    """
    Returns estimated transpiled circuit resource costs (gates, depth).
    """
    num_2q_gates = len(pairs) * cfg.n_steps
    depth = cfg.n_steps * (len(pairs) + cfg.n_qubits)
    return {
        "isa_depth": depth,
        "isa_two_qubit_gates": num_2q_gates,
        "isa_two_qubit_depth": depth // 2,
    }


def build_qiskit_circuit(x, cfg, pairs, n_qubits):
    """
    Constructs a Qiskit circuit for a single input vector x.
    """
    qc = QuantumCircuit(n_qubits)
    # State preparation: RY rotations matching feature dimensions
    for i in range(min(n_qubits, len(x))):
        qc.ry(np.pi * float(x[i]), i)

    # Trotterized TFIM Evolution
    for _ in range(cfg.n_steps):
        # ZZ coupling term exp(-i J_ij dt Z_i Z_j)
        for ci, cj, Jij in pairs:
            # IsingZZ rotation angle is 2 * Jij * dt
            qc.rzz(2.0 * Jij * cfg.dt, ci, cj)
        # Transverse field X term exp(-i h dt X_i)
        for i in range(n_qubits):
            qc.rx(2.0 * cfg.h * cfg.dt, i)

    # Measure all qubits
    qc.measure_all()
    return qc


def run_reservoir_qiskit(
    X, cfg, pairs, backend, n_qubits, shots=4096, optimization_level=1
):
    """
    Constructs the circuits and executes them on the selected qBraid device backend.
    """
    circuits = []
    for x in X:
        qc = build_qiskit_circuit(x, cfg, pairs, n_qubits)
        circuits.append(qc)

    print(f"Submitting batch of {len(circuits)} jobs to {backend.id}...")

    # Run the circuits batch
    # The qBraid device expects a circuit (or list of circuits)
    job = backend.run(circuits, shots=shots)

    # Handle both single QbraidJob and lists of QbraidJobs returned by backend.run
    is_list = isinstance(job, list)
    if is_list:
        print(f"Jobs submitted successfully as a batch list of length {len(job)}!")
        job_ids = [j.id for j in job]
        print(f"Job IDs: {job_ids}")
        # Wait for all jobs to finish and merge results
        results = [j.result() for j in job]
    else:
        print(f"Job submitted successfully! Job ID: {job.id}")
        results = [job.result()]

    # Retrieve expectation values (Z and ZZ) from measurement counts
    # Z_i = P(0) - P(1) on qubit i
    # ZZ_ij = P(00) + P(11) - P(01) - P(10) on qubits i, j
    N = len(X)
    out = np.zeros((N, n_qubits + n_qubits * (n_qubits - 1) // 2), dtype=np.float32)

    idx_out = 0
    for r_obj in results:
        # result.data can be a list of experiment results
        for data in r_obj.data:
            counts = data.get_counts()
            total_counts = sum(counts.values())

            # Calculate Z expectation values
            z_vals = []
            for i in range(n_qubits):
                bit_idx = n_qubits - 1 - i
                count_0 = sum(val for key, val in counts.items() if key[bit_idx] == "0")
                count_1 = total_counts - count_0
                z_vals.append((count_0 - count_1) / total_counts)

            # Calculate ZZ expectation values
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

            out[idx_out] = np.array(z_vals + zz_vals, dtype=np.float32)
            idx_out += 1
            if idx_out >= N:
                break

    return out
