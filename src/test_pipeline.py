import sys
sys.path.insert(0, "/home/claude/qem_project/src")
from circuits import random_hardware_efficient_circuit, ideal_expectation, circuit_features
from noise_sim import build_noise_model, measure_expectation_z0, zne_extrapolate
import time

t0 = time.time()
noise_model, params = build_noise_model()
print("noise params:", params)

qc = random_hardware_efficient_circuit(n_qubits=4, depth=3, seed=42)
ideal = ideal_expectation(qc)
feats = circuit_features(qc, seed=42, n_qubits=4, depth=3)
noisy = measure_expectation_z0(qc, noise_model, shots=2000, seed_simulator=1)
zne_val, raw = zne_extrapolate(qc, noise_model, shots=2000, seed_simulator=1)

print("ideal:", ideal)
print("noisy:", noisy)
print("zne:", zne_val, "raw:", raw)
print("features:", feats)
print("elapsed:", time.time() - t0, "s")
