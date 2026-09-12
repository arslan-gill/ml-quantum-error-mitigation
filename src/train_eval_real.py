"""
Train an ML model to correct noisy expectation values, and compare against:
  (a) raw noisy expectation value (no mitigation)
  (b) Zero-Noise Extrapolation (ZNE) -- standard mitigation baseline

Two evaluation regimes:
  1. Random split (train/test circuits drawn from the same distribution of
     qubit counts / depths). This is the "easy" generalization test.
  2. Out-of-distribution (OOD) split: train on n_qubits in {2,3,4,5}, test
     ONLY on n_qubits == 6 (unseen circuit size). This is the "honest" test
     of whether the model learned something about noise structure or just
     memorized the training distribution.

Model: Gradient Boosted Trees regressor (robust, low-data-friendly, doesn't
need feature scaling). Predicts the ideal expectation value directly from
circuit features + noisy measurement + ZNE fold measurements.
"""
import sys
sys.path.insert(0, "/home/claude/qem_project/src")

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA_PATH = "/home/claude/qem_project/data/dataset_real_noise.csv"
RESULTS_PATH = "/home/claude/qem_project/results/metrics_real_noise.json"

FEATURE_COLS = [
    "n_qubits", "depth_param", "circuit_depth", "n_cx",
    "n_single_qubit_gates", "total_gates", "cx_density",
    "noisy_expectation", "fold1", "fold3", "fold5", "zne_expectation",
]
TARGET_COL = "ideal_expectation"


def evaluate(df_train, df_test, label):
    X_train = df_train[FEATURE_COLS].values
    y_train = df_train[TARGET_COL].values
    X_test = df_test[FEATURE_COLS].values
    y_test = df_test[TARGET_COL].values

    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=0
    )
    model.fit(X_train, y_train)
    y_pred_ml = model.predict(X_test)

    noisy_test = df_test["noisy_expectation"].values
    zne_test = df_test["zne_expectation"].values

    metrics = {
        "n_train": len(df_train),
        "n_test": len(df_test),
        "mse_raw_noisy": mean_squared_error(y_test, noisy_test),
        "mse_zne": mean_squared_error(y_test, zne_test),
        "mse_ml": mean_squared_error(y_test, y_pred_ml),
        "mae_raw_noisy": mean_absolute_error(y_test, noisy_test),
        "mae_zne": mean_absolute_error(y_test, zne_test),
        "mae_ml": mean_absolute_error(y_test, y_pred_ml),
    }

    print(f"\n=== {label} ===")
    print(f"n_train={metrics['n_train']}, n_test={metrics['n_test']}")
    print(f"MSE   raw={metrics['mse_raw_noisy']:.5f}  zne={metrics['mse_zne']:.5f}  ml={metrics['mse_ml']:.5f}")
    print(f"MAE   raw={metrics['mae_raw_noisy']:.5f}  zne={metrics['mae_zne']:.5f}  ml={metrics['mae_ml']:.5f}")

    improvement_over_raw = (1 - metrics['mse_ml'] / metrics['mse_raw_noisy']) * 100
    improvement_over_zne = (1 - metrics['mse_ml'] / metrics['mse_zne']) * 100
    print(f"ML improvement over raw noisy: {improvement_over_raw:.1f}% (MSE reduction)")
    print(f"ML improvement over ZNE:       {improvement_over_zne:.1f}% (MSE reduction)")

    metrics["improvement_over_raw_pct"] = improvement_over_raw
    metrics["improvement_over_zne_pct"] = improvement_over_zne

    return metrics, model, y_test, y_pred_ml, noisy_test, zne_test


def main():
    df = pd.read_csv(DATA_PATH)
    all_metrics = {}

    # --- Regime 1: random split ---
    df_train, df_test = train_test_split(df, test_size=0.25, random_state=0)
    m1, model1, y_test1, y_pred1, noisy1, zne1 = evaluate(
        df_train, df_test, "Regime 1: Random split (in-distribution)"
    )
    all_metrics["random_split"] = m1

    # --- Regime 2: OOD split (train on 2-5 qubits, test on 6 qubits) ---
    df_train_ood = df[df["n_qubits"] <= 5]
    df_test_ood = df[df["n_qubits"] == 6]
    m2, model2, y_test2, y_pred2, noisy2, zne2 = evaluate(
        df_train_ood, df_test_ood, "Regime 2: OOD split (train n_qubits<=5, test n_qubits==6)"
    )
    all_metrics["ood_split"] = m2

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save predictions for plotting
    np.savez(
        "/home/claude/qem_project/results/predictions_real_noise.npz",
        y_test1=y_test1, y_pred1=y_pred1, noisy1=noisy1, zne1=zne1,
        y_test2=y_test2, y_pred2=y_pred2, noisy2=noisy2, zne2=zne2,
    )

    # feature importances from the random-split model
    importances = dict(zip(FEATURE_COLS, model1.feature_importances_.tolist()))
    with open("/home/claude/qem_project/results/feature_importances_real_noise.json", "w") as f:
        json.dump(importances, f, indent=2)
    print("\nFeature importances (random-split model):")
    for k, v in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.3f}")

    print(f"\nSaved metrics to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
