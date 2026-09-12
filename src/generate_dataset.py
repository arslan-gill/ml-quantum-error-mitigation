"""
Generate the full dataset of (features, noisy, zne, ideal) for a diverse
set of circuits, varying qubit count and depth so the ML model has to
generalize across circuit sizes -- not just interpolate within one fixed
circuit shape.
"""
import sys
sys.path.insert(0, "/home/claude/qem_project/src")

import time
import json
import numpy as np
import pandas as pd
from circuits import random_hardware_efficient_circuit, ideal_expectation, circuit_features
from noise_sim import build_noise_model, measure_expectation_z0, zne_extrapolate

OUT_PATH = "/home/claude/qem_project/data/dataset.csv"
SHOTS = 3000

def main():
    t0 = time.time()
    noise_model, noise_params = build_noise_model()
    print("Noise params:", noise_params)

    rows = []
    seed_counter = 0

    qubit_range = [2, 3, 4, 5, 6]
    depth_range = [1, 2, 3, 4, 5, 6, 7, 8]
    n_per_config = 12  # circuits per (n_qubits, depth) combo

    total_configs = len(qubit_range) * len(depth_range)
    config_i = 0

    for n_qubits in qubit_range:
        for depth in depth_range:
            config_i += 1
            for _ in range(n_per_config):
                seed = seed_counter
                seed_counter += 1

                qc = random_hardware_efficient_circuit(n_qubits, depth, seed)
                ideal = ideal_expectation(qc)
                feats = circuit_features(qc, seed, n_qubits, depth)

                noisy = measure_expectation_z0(
                    qc, noise_model, shots=SHOTS, seed_simulator=seed
                )
                zne_val, raw_folds = zne_extrapolate(
                    qc, noise_model, scale_factors=(1, 3, 5),
                    shots=SHOTS, seed_simulator=seed
                )

                row = dict(feats)
                row["noisy_expectation"] = noisy
                row["zne_expectation"] = zne_val
                row["fold1"] = raw_folds[1]
                row["fold3"] = raw_folds[3]
                row["fold5"] = raw_folds[5]
                row["ideal_expectation"] = ideal
                rows.append(row)

            if config_i % 5 == 0 or config_i == total_configs:
                elapsed = time.time() - t0
                print(f"config {config_i}/{total_configs} done, "
                      f"{len(rows)} circuits so far, elapsed {elapsed:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    with open("/home/claude/qem_project/data/noise_params.json", "w") as f:
        json.dump(noise_params, f, indent=2)

    print(f"Saved {len(df)} rows to {OUT_PATH}")
    print(f"Total time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
