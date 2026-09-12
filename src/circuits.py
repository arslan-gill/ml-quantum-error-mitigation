"""
Benchmark circuit generation for the QEM-ML project.

We use a family of circuits whose ideal (noiseless) expectation value of a
fixed observable (Z on qubit 0) can be computed exactly via statevector
simulation. Varying circuit structure (depth, number of qubits, gate
composition, entangling structure) gives us a diverse dataset of
(circuit_features, noisy_expectation, ideal_expectation) triples.

Circuit family: randomized layers of single-qubit rotations (RX, RY, RZ)
interleaved with CX entangling gates, following a hardware-efficient ansatz
pattern common in variational quantum algorithms (VQE/QAOA-style circuits).
This is a realistic stand-in for the kinds of circuits QEM is actually used
on in practice.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp


def random_hardware_efficient_circuit(n_qubits, depth, seed):
    """
    Build a hardware-efficient-ansatz-style circuit with `depth` layers.
    Each layer = single-qubit rotations on every qubit + a ladder of CX gates.
    Returns the circuit (no measurement) and its parameter seed for
    reproducibility.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n_qubits)

    for layer in range(depth):
        # single-qubit rotation layer
        for q in range(n_qubits):
            gate_choice = rng.integers(0, 3)
            angle = rng.uniform(0, 2 * np.pi)
            if gate_choice == 0:
                qc.rx(angle, q)
            elif gate_choice == 1:
                qc.ry(angle, q)
            else:
                qc.rz(angle, q)
        # entangling layer (linear ladder of CX), skip on last layer sometimes
        if n_qubits > 1:
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)

    return qc


def ideal_expectation(qc, observable="Z0"):
    """Exact noiseless expectation value via statevector simulation."""
    n = qc.num_qubits
    sv = Statevector.from_instruction(qc)
    if observable == "Z0":
        pauli_str = "I" * (n - 1) + "Z"
    else:
        raise ValueError("unsupported observable")
    op = SparsePauliOp(pauli_str)
    return float(np.real(sv.expectation_value(op)))


def circuit_features(qc, seed, n_qubits, depth):
    """
    Extract structural features of a circuit for use as ML input.
    These are features an ML model could realistically access without
    knowing the ideal answer: gate counts, depth, qubit count, and simple
    proxies for how much noise a circuit is likely to accumulate.
    """
    ops = qc.count_ops()
    n_cx = ops.get("cx", 0)
    n_single = sum(v for k, v in ops.items() if k != "cx")
    circuit_depth = qc.depth()

    return {
        "seed": seed,
        "n_qubits": n_qubits,
        "depth_param": depth,
        "circuit_depth": circuit_depth,
        "n_cx": n_cx,
        "n_single_qubit_gates": n_single,
        "total_gates": n_cx + n_single,
        "cx_density": n_cx / max(1, n_cx + n_single),
    }
