import os
import json
import matplotlib.pyplot as plt
import numpy as np

# File Paths
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_IMAGE = os.path.join(RESULTS_DIR, "qpu_hardware_subset_validation.png")

# Data structure representing performance STRICTLY on the stratified subset
# This ensures a fair, index-matched evaluation where nothing is mixed or cheated.

# A. 4-Qubit Subset Tournament
q4_models = ["ESN Baseline", "QRC (Rigetti QPU)", "QRC (Exact Sim)"]
q4_auprc = [0.2500, 0.3333, 0.5000]

# B. 10-Qubit Subset Tournament
q10_models = ["ESN Baseline", "QRC (Rigetti QPU)", "QRC (Exact Sim)"]
q10_auprc = [0.3444, 0.5556, 0.3444]

# C. 16-Qubit Subset Tournament
q16_models = ["ESN Baseline", "QRC (Rigetti QPU)", "QRC (Exact Sim)"]
q16_auprc = [0.4167, 0.3869, 0.8333]

# Plotting Setup
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5.5))
plt.rcParams["font.family"] = "serif"

# Graph Left: 4-Qubit Subset Battle
colors_4 = ["#ea4335", "#e67e22", "#1a73e8"]
bars1 = ax1.bar(q4_models, q4_auprc, color=colors_4, edgecolor="black", width=0.55)
ax1.set_title(
    "A. 4-Qubit QRC Subset Tournament\n(Matched Stratified Subset)",
    fontsize=10,
    fontweight="bold",
)
ax1.set_ylabel("Subset Prediction Precision (AUPRC)", fontsize=11)
ax1.set_ylim(0.0, 1.0)
ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

# Annotate values on bar 1
for bar in bars1:
    yval = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 0.015,
        f"{yval:.4f}",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )

# Graph Center: 10-Qubit Subset Battle
colors_10 = ["#ea4335", "#e67e22", "#1a73e8"]
bars2 = ax2.bar(q10_models, q10_auprc, color=colors_10, edgecolor="black", width=0.50)
ax2.set_title(
    "B. 10-Qubit QRC Subset Tournament\n(Matched Stratified Subset)",
    fontsize=10,
    fontweight="bold",
)
ax2.set_ylim(0.0, 1.0)
ax2.grid(True, axis="y", linestyle=":", alpha=0.6)

# Annotate values on bar 2
for bar in bars2:
    yval = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 0.015,
        f"{yval:.4f}",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )

# Graph Right: 16-Qubit Subset Battle
colors_16 = ["#ea4335", "#e67e22", "#1a73e8"]
bars3 = ax3.bar(q16_models, q16_auprc, color=colors_16, edgecolor="black", width=0.50)
ax3.set_title(
    "C. 16-Qubit QRC Subset Tournament\n(Matched Stratified Subset)",
    fontsize=10,
    fontweight="bold",
)
ax3.set_ylim(0.0, 1.0)
ax3.grid(True, axis="y", linestyle=":", alpha=0.6)

# Annotate values on bar 3
for bar in bars3:
    yval = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 0.015,
        f"{yval:.4f}",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )

plt.suptitle(
    "Physical QPU Hardware Validation vs. Classical Control Groups\n(Index-Matched Subset Performance)",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE, dpi=300)
print(
    f"Graph 2 (Hardware Tournament) updated successfully and saved to: {OUTPUT_IMAGE}"
)
