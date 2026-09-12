# ML for Quantum Error Mitigation

A machine-learning correction model for noisy quantum circuit expectation
values, benchmarked against Zero-Noise Extrapolation (ZNE) under two noise
models: an invented uniform depolarizing model, and a real per-qubit noise
model built from a frozen calibration snapshot of IBM's 27-qubit
`ibmq_mumbai` device.

📄 **[Full write-up (PDF)](results/QEM_ML_Report.md)** — methodology, all figures, and references.

## Headline result

Gradient-Boosted-Trees correction reduces MSE vs. the ideal expectation
value by:

| | vs. raw noisy | vs. ZNE |
|---|---|---|
| Invented noise, random split | 77.4% | 49.0% |
| Invented noise, OOD split (unseen 6-qubit circuits) | 67.7% | 39.9% |
| Real ibmq_mumbai noise, random split | 74.3% | 35.1% |
| Real ibmq_mumbai noise, OOD split | 74.6% | 42.1% |

## Repo layout

```
src/
  circuits.py                 benchmark circuit generation + exact ideal expectation values
  noise_sim.py                invented uniform noise model, gate folding, ZNE
  real_noise.py                real ibmq_mumbai-calibrated noise model (FakeMumbaiV2)
  generate_dataset.py         builds the 480-circuit dataset under invented noise
  generate_dataset_real.py    builds the 480-circuit dataset under real-calibration noise (chunked by --n_qubits)
  train_eval.py                trains + evaluates the ML model (invented-noise dataset)
  train_eval_real.py          trains + evaluates the ML model (real-noise dataset)
  make_figures.py             figures for the invented-noise results
  make_comparison_figures.py  invented-vs-real comparison figures
  make_report.js              generates the Word report (results/QEM_ML_Report.docx)
  test_pipeline.py            small smoke test of the invented-noise pipeline

data/
  dataset.csv                  480 circuits, invented noise model
  dataset_real_noise.csv      480 circuits, real ibmq_mumbai-calibrated noise model
  noise_params.json           invented noise model parameters
  real_noise_calibration.json  real per-qubit/per-edge calibration data used

results/
  metrics*.json, feature_importances*.json, predictions*.npz
  QEM_ML_Report.docx / .pdf    full report

figures/
  fig1-fig6                    all report figures
```

## Reproducing

```bash
pip install qiskit qiskit-aer qiskit-ibm-runtime scikit-learn matplotlib pandas --break-system-packages

# Invented uniform noise model
python3 src/generate_dataset.py
python3 src/train_eval.py
python3 src/make_figures.py

# Real ibmq_mumbai-calibrated noise model (chunked; ~90s per qubit count)
for n in 2 3 4 5 6; do python3 src/generate_dataset_real.py --n_qubits $n; done
python3 src/train_eval_real.py
python3 src/make_comparison_figures.py

# Word report (requires Node + the `docx` npm package)
npm install docx
node src/make_report.js
```

## Method summary

- **Circuits**: hardware-efficient-ansatz style (random single-qubit
  RX/RY/RZ layers + linear CX ladder), 2-6 qubits, depth 1-8, 12 seeds per
  (qubits, depth) config -> 480 circuits. Ideal `<Z_0>` computed exactly via
  statevector simulation.
- **Noise models**: (1) invented uniform depolarizing + readout error; (2)
  real per-qubit/per-edge error rates from a frozen `FakeMumbaiV2`
  (`ibmq_mumbai`) calibration snapshot, mapped onto a real 6-qubit linear
  chain in the device's coupling map with zero swap overhead.
- **ZNE baseline**: global unitary folding at scale factors 1/3/5, linear
  (Richardson) extrapolation to the zero-noise limit.
- **ML model**: Gradient Boosted Trees regressor trained on circuit
  structural features + noisy/folded measurements + the ZNE point estimate,
  predicting the ideal expectation value directly.
- **Evaluation**: random train/test split (in-distribution) and an
  out-of-distribution split (train on 2-5 qubits, test only on unseen
  6-qubit circuits), under both noise models.

## Known limitations

- No circuits were executed on live quantum hardware; all noisy execution
  is Qiskit Aer simulation (with either invented or real-calibration-snapshot
  noise parameters).
- The real-calibration noise model uses a frozen historical snapshot, not a
  live pull from a currently-running device.
- Single observable (`<Z_0>`) and a single circuit family; generalization to
  other observables or circuit families (e.g. QAOA cost Hamiltonians) is
  untested.
- No direct comparison to Clifford Data Regression (CDR/vnCDR) or
  Probabilistic Error Cancellation.

See the [full report](results/QEM_ML_Report.pdf), Section 6, for the full discussion.
