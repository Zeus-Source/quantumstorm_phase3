import os
import json
import numpy as np
import matplotlib.pyplot as plt

# File Paths
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_IMAGE = os.path.join(RESULTS_DIR, "qrc_theoretical_scaling_extrapolation.png")

# 1. Existing verified classically emulated data points
qubits = np.array([4, 6, 10, 16], dtype=np.float32)
auprc = np.array([0.2402, 0.2946, 0.3850, 0.4561], dtype=np.float32)

# 2. Fit a logarithmic curve: AUPRC = a * ln(N) + b
# Since predicting storm is bounded by 1.0 (perfect precision), logarithmic scaling fits well
# as performance gains exhibit diminishing returns (saturation) as qubits grow.
coefs = np.polyfit(np.log(qubits), auprc, 1)  # fits y = a*log(x) + b
a, b = coefs[0], coefs[1]

# 3. Extrapolate up to the top Rigetti QPU machine (Rigetti Cepheus-1-108q has 107 active qubits!)
max_rigetti_qubits = 107
extrap_qubits = np.arange(4, max_rigetti_qubits + 1, dtype=np.float32)
extrap_auprc = a * np.log(extrap_qubits) + b

# Cap theoretical prediction at 0.95 to account for irreducible dataset chaos noise (entropy limits)
extrap_auprc = np.minimum(extrap_auprc, 0.95)

# Calculate expected AUPRC at key milestone qubit targets
q80_val = a * np.log(80) + b
q107_val = a * np.log(107) + b

# Load ESN baseline for reference
esn_auprc = 0.4101

# Plotting Setup
plt.figure(figsize=(10, 6.5))
plt.rcParams["font.family"] = "serif"

# Plot the extrapolation curve
plt.plot(
    extrap_qubits,
    extrap_auprc,
    "--",
    color="#202124",
    linewidth=2,
    label="Logarithmic Extrapolation Fit ($a \\cdot \\ln(N) + b$)",
)

# Plot the validated simulator data points (as blue dots)
plt.scatter(
    qubits,
    auprc,
    color="#1a73e8",
    s=100,
    zorder=5,
    edgecolor="black",
    label="Classically Emulated Benchmarks (4, 6, 10, 16 Qubits)",
)

# Plot reference line for ESN
plt.axhline(
    y=esn_auprc,
    color="#ea4335",
    linestyle="--",
    linewidth=1.8,
    label=f"ESN Baseline (AUPRC = {esn_auprc:.4f})",
)

# Highlight milestones on Rigetti hardware
plt.scatter(
    [80, 107],
    [q80_val, q107_val],
    color="#e28743",
    marker="X",
    s=150,
    zorder=6,
    edgecolor="black",
    label="Target Milestones on Rigetti QPUs",
)

# Annotation box for Rigetti Cepheus limit
plt.annotate(
    f"Rigetti Cepheus-1-108q Target\n({max_rigetti_qubits} Qubits)\nPredicted AUPRC: ~{q107_val:.4f}",
    xy=(max_rigetti_qubits, q107_val),
    xytext=(max_rigetti_qubits - 45, q107_val - 0.22),
    arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=8),
    fontsize=10,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.5", fc="#fdf6e3", ec="#e67e22", lw=1.5),
)

# Annotate milestone at 80 qubits
plt.annotate(
    f"80-Qubits Milestone\nPredicted AUPRC: ~{q80_val:.4f}",
    xy=(80, q80_val),
    xytext=(80 - 45, q80_val + 0.12),
    arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=8),
    fontsize=9.5,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#70757a", lw=1),
)

# Format zones
plt.fill_between(
    extrap_qubits,
    esn_auprc,
    extrap_auprc,
    where=(extrap_auprc > esn_auprc),
    color="#1a73e8",
    alpha=0.06,
    label="Predicted Quantum Advantage Growth Zone",
)

# Titles & Axes
plt.title(
    "QRC Performance Extrapolation vs. Qubit Capacity Scaling\n(Predicting Theoretical Limits on Rigetti Hardware Platforms)",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
plt.xlabel("Number of Active Qubits ($N$)", fontsize=11)
plt.ylabel("Extreme Event Prediction Precision (AUPRC)", fontsize=11)
plt.xlim(2, 115)
plt.ylim(0.0, 1.0)
plt.grid(True, linestyle=":", alpha=0.5)
plt.legend(
    loc="lower right",
    fontsize=9.5,
    frameon=True,
    facecolor="white",
    edgecolor="#dadce0",
)

# Save Plot
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300)
print(f"Graph 3 (Theoretical Extrapolation) saved to: {OUTPUT_IMAGE}")
