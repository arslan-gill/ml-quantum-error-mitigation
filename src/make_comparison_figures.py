import json
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = "/home/claude/qem_project/figures"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

with open("/home/claude/qem_project/results/metrics.json") as f:
    m_invented = json.load(f)
with open("/home/claude/qem_project/results/metrics_real_noise.json") as f:
    m_real = json.load(f)
with open("/home/claude/qem_project/data/real_noise_calibration.json") as f:
    calib = json.load(f)

# ---- Figure: side-by-side MSE, invented vs real noise model ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
regimes = ["random_split", "ood_split"]
regime_labels = ["Random split", "OOD split (test n=6)"]
methods = ["mse_raw_noisy", "mse_zne", "mse_ml"]
method_labels = ["Raw noisy", "ZNE", "ML-corrected"]
colors = ["#c0392b", "#e0a020", "#2e7d32"]

for ax, m, title in zip(axes, [m_invented, m_real], ["Invented uniform depolarizing noise", "Real IBM Mumbai calibration snapshot"]):
    x = np.arange(len(regimes))
    width = 0.25
    for i, (mk, ml, c) in enumerate(zip(methods, method_labels, colors)):
        vals = [m[r][mk] for r in regimes]
        ax.bar(x + (i - 1) * width, vals, width, label=ml, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(regime_labels)
    ax.set_ylabel("MSE vs. ideal expectation value")
    ax.set_title(title, fontsize=10.5)
axes[0].legend(fontsize=9)
fig.suptitle("ML correction beats ZNE under both an invented and a real-hardware-calibrated noise model", y=1.03)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig5_invented_vs_real_mse.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure: real per-qubit calibration data used ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
qubits = [str(r["qubit"]) for r in calib["qubit_calibration"]]
readout = [r["readout_error"] * 100 for r in calib["qubit_calibration"]]
t1 = [r["t1_us"] for r in calib["qubit_calibration"]]

axes[0].bar(qubits, readout, color="#8e44ad")
axes[0].set_xlabel("Physical qubit (ibmq_mumbai)")
axes[0].set_ylabel("Readout error (%)")
axes[0].set_title("Real per-qubit readout error")

axes[1].bar(qubits, t1, color="#2980b9")
axes[1].set_xlabel("Physical qubit (ibmq_mumbai)")
axes[1].set_ylabel("T1 (\u03bcs)")
axes[1].set_title("Real per-qubit T1 relaxation time")

fig.suptitle("Real calibration data used for the noise model (frozen ibmq_mumbai snapshot)", y=1.03)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig6_real_calibration_data.png", bbox_inches="tight")
plt.close(fig)

print("Saved comparison figures.")
print(json.dumps({"invented": m_invented, "real": m_real}, indent=2))
