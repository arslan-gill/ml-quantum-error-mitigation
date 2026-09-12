"""
Generate the dataset using the REAL-hardware-calibrated noise model
(FakeMumbaiV2 snapshot) instead of the invented uniform depolarizing model.
Same circuit family, same (n_qubits, depth) grid, same seeds as the
original dataset, so the two are directly comparable.
"""
import sys
sys.path.insert(0, "/home/claude/qem_project/src")

import os
import time
import json
import argparse
import pandas as pd
from circuits import random_hardware_efficient_circuit, ideal_expectation, circuit_features
from real_noise import (
    get_real_backend_simulator, measure_expectation_z0_real,
    zne_extrapolate_real, describe_chain_noise, BACKEND_NAME, QUBIT_CHAIN
)

OUT_PATH = "/home/claude/qem_project/data/dataset_real_noise.csv"
SHOTS = 3000

DEPTH_RANGE = [1, 2, 3, 4, 5, 6, 7, 8]
N_PER_CONFIG = 12
QUBIT_RANGE = [2, 3, 4, 5, 6]  # full grid, one value processed per invocation


def seed_offset(n_qubits):
    """Deterministic seed offset per n_qubits so chunked runs match a single full run."""
    idx = QUBIT_RANGE.index(n_qubits)
    return idx * len(DEPTH_RANGE) * N_PER_CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_qubits", type=int, required=True)
    args = parser.parse_args()
    n_qubits = args.n_qubits

    t0 = time.time()
    sim, backend, chain = get_real_backend_simulator()

    if not os.path.exists("/home/claude/qem_project/data/real_noise_calibration.json"):
        rows_noise, edges_noise = describe_chain_noise(backend, chain)
        with open("/home/claude/qem_project/data/real_noise_calibration.json", "w") as f:
            json.dump({"backend": BACKEND_NAME, "chain": chain,
                       "qubit_calibration": rows_noise, "cx_calibration": edges_noise}, f, indent=2)

    rows = []
    seed_counter = seed_offset(n_qubits)

    for depth in DEPTH_RANGE:
        for _ in range(N_PER_CONFIG):
            seed = seed_counter
            seed_counter += 1

            qc = random_hardware_efficient_circuit(n_qubits, depth, seed)
            ideal = ideal_expectation(qc)
            feats = circuit_features(qc, seed, n_qubits, depth)

            noisy = measure_expectation_z0_real(
                qc, sim, chain, shots=SHOTS, seed_simulator=seed
            )
            zne_val, raw_folds = zne_extrapolate_real(
                qc, sim, chain, scale_factors=(1, 3, 5),
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

        print(f"n_qubits={n_qubits} depth={depth} done, "
              f"{len(rows)} circuits so far, elapsed {time.time()-t0:.1f}s")

    df = pd.DataFrame(rows)
    header = not os.path.exists(OUT_PATH)
    df.to_csv(OUT_PATH, mode="a", header=header, index=False)
    print(f"Appended {len(df)} rows to {OUT_PATH}. Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
