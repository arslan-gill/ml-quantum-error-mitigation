"""
Noisy simulation + Zero-Noise Extrapolation (ZNE) baseline.

Noise model: depolarizing error on single- and two-qubit gates plus
readout (measurement) error, with parameters in the range reported for
real superconducting devices (e.g., IBM Quantum backends circa 2023-2024):
  - 1q depolarizing error ~ 1e-4 - 1e-3
  - 2q (CX) depolarizing error ~ 5e-3 - 2e-2
  - readout error ~ 1e-2 - 3e-2
This is intentionally realistic-but-simplified: a single global noise
model rather than per-qubit calibration data, since we don't have queue
access to real hardware for this project. This limitation is noted
explicitly in the writeup.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    ReadoutError,
)
from qiskit.quantum_info import SparsePauliOp

SHOTS = 4000


def build_noise_model(p1=5e-4, p2=1.2e-2, p_readout=2e-2, seed=None):
    """Construct a depolarizing + readout noise model."""
    noise_model = NoiseModel()

    error_1 = depolarizing_error(p1, 1)
    error_2 = depolarizing_error(p2, 2)

    noise_model.add_all_qubit_quantum_error(
        error_1, ["rx", "ry", "rz", "id", "sx", "x"]
    )
    noise_model.add_all_qubit_quantum_error(error_2, ["cx"])

    ro_error = ReadoutError(
        [[1 - p_readout, p_readout], [p_readout, 1 - p_readout]]
    )
    noise_model.add_all_qubit_readout_error(ro_error)

    return noise_model, dict(p1=p1, p2=p2, p_readout=p_readout)


def fold_circuit_global(qc, scale_factor):
    """
    Global unitary folding for ZNE: replaces circuit U with U (U^dagger U)^n
    to scale the effective noise by (2n+1) without changing the ideal
    output. scale_factor must be an odd integer (1, 3, 5, ...).
    """
    assert scale_factor % 2 == 1 and scale_factor >= 1
    n_folds = (scale_factor - 1) // 2
    folded = qc.copy()
    for _ in range(n_folds):
        folded = folded.compose(qc.inverse())
        folded = folded.compose(qc)
    return folded


def measure_expectation_z0(qc, noise_model, shots=SHOTS, seed_simulator=None):
    """
    Estimate <Z_0> under noise by sampling in the computational basis
    (equivalent to measuring Z on qubit 0 directly since Z is diagonal).
    """
    n = qc.num_qubits
    meas_qc = qc.copy()
    meas_qc.measure_all()

    sim = AerSimulator(noise_model=noise_model)
    tqc = transpile(meas_qc, sim, optimization_level=0)
    result = sim.run(tqc, shots=shots, seed_simulator=seed_simulator).result()
    counts = result.get_counts()

    total = sum(counts.values())
    exp_val = 0.0
    for bitstring, count in counts.items():
        # qiskit bit ordering: rightmost char = qubit 0
        bit0 = bitstring.replace(" ", "")[-1]
        z = 1 if bit0 == "0" else -1
        exp_val += z * count
    return exp_val / total


def zne_extrapolate(qc, noise_model, scale_factors=(1, 3, 5), shots=SHOTS, seed_simulator=None):
    """
    Standard Richardson (linear) zero-noise extrapolation baseline.
    Measures <Z0> at several noise scale factors via global folding,
    then fits a line and extrapolates to zero noise (x=0).
    Returns (zne_estimate, raw_measurements_dict).
    """
    xs = []
    ys = []
    raw = {}
    for s in scale_factors:
        folded = fold_circuit_global(qc, s)
        val = measure_expectation_z0(
            folded, noise_model, shots=shots, seed_simulator=seed_simulator
        )
        xs.append(s)
        ys.append(val)
        raw[s] = val

    # linear fit y = a*x + b, extrapolate to x=0 -> b
    coeffs = np.polyfit(xs, ys, deg=1)
    zne_estimate = coeffs[-1]  # intercept
    return float(zne_estimate), raw
