"""
Real-hardware-calibrated noise model.

Instead of inventing a single global depolarizing rate, this module builds
the noise model from a frozen historical calibration snapshot of a real
IBM Quantum device, shipped with Qiskit for exactly this kind of realistic
offline testing: qiskit_ibm_runtime.fake_provider.FakeMumbaiV2, which
mirrors ibmq_mumbai (27-qubit Falcon r4 processor). This gives us:

  - per-qubit T1/T2 relaxation times
  - per-qubit, per-gate error rates (not a single global number)
  - per-qubit readout error (heterogeneous, not uniform)
  - the device's real native basis gate set (rz, sx, x, cx) and coupling map

We select a linear chain of 6 physically-connected qubits from the real
27-qubit coupling map (found via a longest-simple-path search), so our
hardware-efficient-ansatz circuits (which use a linear CX ladder) map onto
real, adjacent hardware qubits with NO swap-routing overhead -- keeping
"logical qubit 0" pinned to a fixed physical qubit throughout.

Caveat (see report Limitations): this is a frozen, historical calibration
snapshot, not a live pull from current hardware -- and no circuits are
executed on real hardware in this project. But every error rate below is
a real number IBM's calibration procedure measured on a real device, not
an invented global constant.
"""

from qiskit_ibm_runtime.fake_provider import FakeMumbaiV2
from qiskit_aer import AerSimulator
from qiskit import transpile

BACKEND_NAME = "FakeMumbaiV2 (real historical calibration snapshot of ibmq_mumbai, 27-qubit Falcon r4)"

# Longest simple path found in ibmq_mumbai's real coupling map (see notebook /
# exploration log): qubits are physically adjacent, so our linear CX-ladder
# circuits transpile with zero swap overhead.
QUBIT_CHAIN = [13, 12, 10, 7, 4, 1]


def get_real_backend_simulator():
    """Return (simulator, backend, chain) built from the real calibration snapshot."""
    backend = FakeMumbaiV2()
    simulator = AerSimulator.from_backend(backend)
    return simulator, backend, QUBIT_CHAIN


def describe_chain_noise(backend, chain):
    """Return a small table of the real per-qubit / per-edge error rates used."""
    props = backend.properties()
    rows = []
    for q in chain:
        rows.append({
            "qubit": q,
            "readout_error": props.readout_error(q),
            "t1_us": props.t1(q) * 1e6,
            "t2_us": props.t2(q) * 1e6,
        })
    edges = []
    for a, b in zip(chain, chain[1:]):
        try:
            err = props.gate_error("cx", [a, b])
        except Exception:
            err = props.gate_error("cx", [b, a])
        edges.append({"qubit_pair": (a, b), "cx_error": err})
    return rows, edges


def transpile_to_chain(qc, simulator, chain):
    """Transpile a logical circuit onto the fixed physical chain, no routing needed."""
    n = qc.num_qubits
    # optimization_level MUST be 0 here: higher levels run gate-cancellation
    # passes that algebraically simplify away the redundant U^dagger U pairs
    # ZNE folding inserts, destroying the noise-scaling effect before the
    # circuit ever reaches the noisy simulator.
    return transpile(
        qc, backend=simulator, initial_layout=chain[:n],
        optimization_level=0, seed_transpiler=0,
    )


def measure_expectation_z0_real(qc, simulator, chain, shots, seed_simulator=None):
    """
    Transpile onto the real chain, measure only the physical qubit that holds
    logical qubit 0 (chain[0]), and return the estimated <Z_0>.
    """
    from qiskit import ClassicalRegister
    n = qc.num_qubits
    tqc = transpile_to_chain(qc, simulator, chain)
    creg = ClassicalRegister(1, "c")
    tqc.add_register(creg)
    tqc.measure(chain[0], creg[0])

    result = simulator.run(tqc, shots=shots, seed_simulator=seed_simulator).result()
    counts = result.get_counts()

    total = sum(counts.values())
    exp_val = 0.0
    for bitstring, count in counts.items():
        bit = bitstring.replace(" ", "")[-1]
        z = 1 if bit == "0" else -1
        exp_val += z * count
    return exp_val / total


def zne_extrapolate_real(qc, simulator, chain, scale_factors=(1, 3, 5), shots=3000, seed_simulator=None):
    """ZNE baseline using the real-noise simulator: fold the logical circuit,
    then transpile each folded version onto the real chain and measure."""
    import numpy as np
    from noise_sim import fold_circuit_global

    xs, ys, raw = [], [], {}
    for s in scale_factors:
        folded = fold_circuit_global(qc, s)
        val = measure_expectation_z0_real(folded, simulator, chain, shots=shots, seed_simulator=seed_simulator)
        xs.append(s)
        ys.append(val)
        raw[s] = val

    coeffs = np.polyfit(xs, ys, deg=1)
    zne_estimate = coeffs[-1]
    return float(zne_estimate), raw
