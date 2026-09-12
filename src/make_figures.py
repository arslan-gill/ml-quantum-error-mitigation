import sys
sys.path.insert(0, "/home/claude/qem_project/src")

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = "/home/claude/qem_project/figures"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

data = np.load("/home/claude/qem_project/results/predictions.npz")
with open("/home/claude/qem_project/results/metrics.json") as f:
    metrics = json.load(f)
with open("/home/claude/qem_project/results/feature_importances.json") as f:
    importances = json.load(f)

# ---- Figure 1: Parity plots (predicted vs ideal) for raw/zne/ml, random split ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True, sharey=True)
y_test = data["y_test1"]
for ax, (pred, name) in zip(
    axes, [(data["noisy1"], "Raw noisy"), (data["zne1"], "ZNE"), (data["y_pred1"], "ML-corrected")]
):
    ax.scatter(y_test, pred, s=14, alpha=0.55, color="#3b6ea5")
    lims = [-1.05, 1.05]
    ax.plot(lims, lims, "k--", linewidth=1, alpha=0.6)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Ideal $\\langle Z_0 \\rangle$")
    ax.set_title(name)
axes[0].set_ylabel("Predicted / measured $\\langle Z_0 \\rangle$")
fig.suptitle("Predicted vs. ideal expectation value (random split, test set)", y=1.03)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig1_parity_random_split.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: MSE bar comparison, both regimes ----
fig, ax = plt.subplots(figsize=(7, 4.5))
regimes = ["random_split", "ood_split"]
labels = ["Random split\n(in-distribution)", "OOD split\n(train n<=5, test n=6)"]
methods = ["mse_raw_noisy", "mse_zne", "mse_ml"]
method_labels = ["Raw noisy", "ZNE", "ML-corrected"]
colors = ["#c0392b", "#e0a020", "#2e7d32"]

x = np.arange(len(regimes))
width = 0.25
for i, (m, ml, c) in enumerate(zip(methods, method_labels, colors)):
    vals = [metrics[r][m] for r in regimes]
    ax.bar(x + (i - 1) * width, vals, width, label=ml, color=c)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("MSE vs. ideal expectation value")
ax.set_title("Error mitigation performance: raw vs. ZNE vs. ML-corrected")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig2_mse_comparison.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure 3: Feature importance ----
fig, ax = plt.subplots(figsize=(7, 4.5))
items = sorted(importances.items(), key=lambda x: x[1])
names = [k for k, v in items]
vals = [v for k, v in items]
ax.barh(names, vals, color="#3b6ea5")
ax.set_xlabel("Feature importance (Gradient Boosting)")
ax.set_title("Which features drive the ML correction?")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig3_feature_importance.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure 4: Error distribution (residuals) ----
fig, ax = plt.subplots(figsize=(7, 4.5))
resid_raw = data["noisy1"] - y_test
resid_zne = data["zne1"] - y_test
resid_ml = data["y_pred1"] - y_test
bins = np.linspace(-0.3, 0.3, 40)
ax.hist(resid_raw, bins=bins, alpha=0.5, label="Raw noisy", color="#c0392b")
ax.hist(resid_zne, bins=bins, alpha=0.5, label="ZNE", color="#e0a020")
ax.hist(resid_ml, bins=bins, alpha=0.5, label="ML-corrected", color="#2e7d32")
ax.axvline(0, color="black", linewidth=1, linestyle="--")
ax.set_xlabel("Residual (estimate - ideal)")
ax.set_ylabel("Count")
ax.set_title("Error distribution across methods (random split, test set)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/fig4_residual_distribution.png", bbox_inches="tight")
plt.close(fig)

print("Saved 4 figures to", FIG_DIR)
