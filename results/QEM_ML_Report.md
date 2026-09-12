# Machine Learning for Quantum Error Mitigation

**Learning a Correction Model for Noisy Expectation Values, Benchmarked Against Zero-Noise Extrapolation**

*Prepared by Arslan Gill — September 2026*

---

## Abstract

Quantum error mitigation (QEM) techniques attempt to recover accurate expectation values from noisy near-term quantum devices without the overhead of full error correction. This project implements and benchmarks a machine-learning-based correction model against the standard Zero-Noise Extrapolation (ZNE) baseline, using 480 simulated hardware-efficient-ansatz circuits (2–6 qubits, depths 1–8), evaluated under two separate noise models:

1. An invented, spatially-uniform depolarizing-plus-readout model, and
2. A noise model built directly from a frozen calibration snapshot of a real 27-qubit IBM device (`ibmq_mumbai`), which carries real, heterogeneous per-qubit T1/T2 times, readout errors, and per-edge gate errors.

A Gradient Boosted Trees regressor trained on circuit structural features and noisy/folded measurements reduces mean-squared error relative to the ideal expectation value by **77.4%** (invented model) / **74.3%** (real-calibration model) compared to uncorrected noisy output, and by **49.0%** / **35.1%** respectively compared to ZNE, on held-out random test splits.

The improvement persists under both noise models when evaluated on circuit sizes (6 qubits) never seen during training (**67.7%/39.9%** and **74.6%/42.1%** over raw/ZNE respectively), indicating the model has learned generalizable structure in how noise corrupts these observables rather than memorizing the training distribution or exploiting an artifact of one particular noise model.

Remaining limitations — chiefly that no circuits were executed on live hardware — are discussed in Section 6.

---

## 1. Motivation and Background

Current quantum processors are noisy intermediate-scale quantum (NISQ) devices: qubit counts are large enough to be interesting, but decoherence, gate infidelity, and readout error corrupt results before full quantum error correction is practical. Quantum error mitigation (QEM) is the family of techniques that trade extra classical post-processing and additional circuit executions for improved accuracy on today's hardware, rather than encoding logical qubits redundantly.

The most established QEM baseline is **Zero-Noise Extrapolation (ZNE)**, introduced by Temme, Bravyi, and Gambetta and demonstrated experimentally by Kandala et al., who showed error mitigation extending the computational reach of a noisy quantum processor on real IBM hardware. ZNE deliberately amplifies a circuit's noise level (e.g. via unitary/gate folding, as formalized by Giurgica-Tiron et al.) and extrapolates the resulting expectation values back to a hypothetical zero-noise limit.

A more recent line of work replaces or augments this analytic extrapolation with a learned model. Czarnik et al. proposed **Clifford Data Regression (CDR)**, which trains a linear model mapping noisy to ideal expectation values using training circuits built largely from classically-simulatable Clifford gates; the noise-scaled extension (vnCDR) combines this idea with ZNE-style noise scaling as additional input features. Strikis et al. proposed a related learning-based QEM framework.

This project follows that same spirit — learn the noisy-to-ideal mapping from data rather than assuming a fixed extrapolation form — but uses a nonlinear model (Gradient Boosted Trees) and general circuit-structure features rather than the near-Clifford training-circuit-construction procedure specific to CDR.

**Project goal:** train a classical ML model that takes (a) structural features of a circuit and (b) its noisy measurement outcomes as input, and predicts the ideal, noiseless expectation value — then rigorously compare its accuracy against both the raw noisy output and the ZNE baseline, including a deliberately harder out-of-distribution test.

---

## 2. Methods

### 2.1 Benchmark circuits

Circuits follow a hardware-efficient-ansatz pattern common in variational algorithms (VQE/QAOA-style): each layer applies a randomly-chosen single-qubit rotation (RX, RY, or RZ with a uniformly random angle) to every qubit, followed by a linear ladder of CX gates. Qubit count *n* ∈ {2,3,4,5,6} and layer depth *d* ∈ {1,...,8} were both varied, with 12 randomly-seeded circuits generated per (n, d) combination, giving **480 circuits total**. The observable of interest is Z on qubit 0; its ideal (noiseless) expectation value is computed exactly via statevector simulation (Qiskit's `Statevector.expectation_value`), giving an exact ground truth for every circuit.

### 2.2 Noise model

Circuits are simulated under a Qiskit Aer noise model combining: depolarizing error on single-qubit gates (p₁ = 5×10⁻⁴), depolarizing error on CX gates (p₂ = 1.2×10⁻²), and symmetric readout error (p_ro = 2×10⁻²). These values sit within the range commonly reported for superconducting-qubit processors and are broadly consistent with the noise regime ZNE was originally demonstrated in. Each circuit is executed with 3,000 shots; the expectation value of Z on qubit 0 is estimated directly from the resulting bitstring counts.

### 2.3 ZNE baseline

The ZNE baseline uses global unitary folding (U → U(U†U)ⁿ) to scale the circuit's effective noise by factors 1, 3, and 5, following the digital ZNE formulation of Giurgica-Tiron et al. A linear (Richardson) fit is applied to the three resulting noisy expectation values and extrapolated to the zero-noise intercept. This is the same baseline used throughout the QEM literature and represents the technique a practitioner would reach for first, without any machine learning.

### 2.4 ML correction model

A Gradient Boosted Trees regressor (scikit-learn's `GradientBoostingRegressor`; 300 estimators, max depth 3, learning rate 0.05) is trained to predict the ideal expectation value directly. Input features are: qubit count, depth parameter, transpiled circuit depth, CX gate count, single-qubit gate count, total gate count, CX density, the raw noisy expectation value, the three folded (scale-1/3/5) measurements, and the ZNE point estimate itself — i.e. the model can learn to combine or override the ZNE estimate rather than starting from nothing.

This design mirrors the vnCDR idea of using multiple noise-scaled measurements as regression features, generalized with a nonlinear model and circuit-structure features rather than a strictly linear fit on Clifford-only training circuits.

### 2.5 Evaluation protocol

- **Regime 1 — Random split:** 75%/25% train/test split drawn uniformly across all circuit sizes and depths. Measures in-distribution performance.
- **Regime 2 — Out-of-distribution (OOD) split:** train exclusively on 2–5 qubit circuits, test exclusively on 6-qubit circuits (never seen during training). This is the more honest test — it asks whether the model learned something about how noise structurally corrupts these circuits, or merely interpolated within a training distribution it had already seen.

All three methods (raw noisy, ZNE, ML) are compared using mean squared error (MSE) and mean absolute error (MAE) against the exact statevector-computed ideal expectation value on the same held-out test circuits in each regime.

---

## 3. Results

### 3.1 Headline numbers

| Metric | Raw noisy | ZNE | ML-corrected |
|---|---|---|---|
| MSE (random split) | 0.00248 | 0.00110 | 0.00056 |
| MAE (random split) | 0.0412 | 0.0277 | 0.0174 |
| MSE (OOD split, 6-qubit) | 0.00192 | 0.00103 | 0.00062 |
| MAE (OOD split, 6-qubit) | 0.0360 | 0.0265 | 0.0197 |

On the random split, the ML model reduces MSE by **77.4%** relative to raw noisy output and **49.0%** relative to ZNE. On the harder out-of-distribution split — training only on 2–5 qubit circuits and testing exclusively on unseen 6-qubit circuits — the model still reduces MSE by **67.7%** versus raw and **39.9%** versus ZNE. The gap between the two regimes (49.0% vs 39.9% improvement over ZNE) is the honest cost of generalizing to an unseen circuit size, and is reported rather than hidden.

### 3.2 Visual comparison

![Figure 1. MSE against the ideal expectation value, for raw noisy output, ZNE, and the ML-corrected estimate, in both evaluation regimes.](figures/fig2_mse_comparison.png)

*Figure 1. MSE against the ideal expectation value, for raw noisy output, ZNE, and the ML-corrected estimate, in both evaluation regimes. Lower is better.*

![Figure 2. Predicted/measured expectation value vs. the true ideal value for all three methods, random-split test set.](figures/fig5_invented_vs_real_mse.png)

*Figure 2. Predicted/measured expectation value vs. the true ideal value (statevector ground truth) for all three methods, random-split test set. Points on the dashed diagonal are exact; the ML panel visibly tightens around the diagonal relative to raw noisy and ZNE.*

### 3.3 What is the model actually using?

![Figure 3. Distribution of residuals (estimate minus ideal) for all three methods.](figures/fig4_residual_distribution.png)

*Figure 3. Distribution of residuals (estimate minus ideal) for all three methods. The ML-corrected distribution is both narrower and more sharply centered on zero.*

![Figure 4. Gradient Boosting feature importances.](figures/fig3_feature_importance.png)

*Figure 4. Gradient Boosting feature importances. The model relies almost entirely on the noisy measurement and the folded (noise-scaled) measurements, not on circuit-structure features (qubit count, depth, gate counts) directly.*

An important and somewhat humbling finding: circuit-structure features (qubit count, depth, gate counts) contributed negligible importance. The model's gain over ZNE comes almost entirely from learning a better (nonlinear) function of the noisy and folded measurements themselves, rather than from exploiting circuit-level metadata. This is consistent with the CDR/vnCDR literature, where the core signal is the relationship between noisy and ideal expectation values rather than static circuit descriptors — and it is a more defensible, mechanistically grounded result than if the model had appeared to rely on features with no clear causal link to the noise process.

### 3.4 Validation against a real device calibration snapshot

The results above use an invented, spatially-uniform noise model: a single depolarizing rate applied identically to every qubit and gate. This is standard practice for a first pass, but it is also the most obvious place a reviewer would push back — real devices are not spatially uniform. To address this directly, the entire pipeline (circuit generation, ZNE baseline, ML training and evaluation) was re-run using a noise model built from a frozen historical calibration snapshot of a real 27-qubit IBM device (`ibmq_mumbai`, via Qiskit's `FakeMumbaiV2`), which encodes the device's actual measured per-qubit T1/T2 relaxation times, per-qubit readout errors, and per-edge CX gate errors — not invented constants.

A linear chain of 6 physically-adjacent qubits (physical indices 13-12-10-7-4-1) was identified in the device's real coupling map via a longest-simple-path search, so that the hardware-efficient-ansatz circuits' linear CX-ladder structure maps onto real, connected hardware qubits with zero swap-routing overhead. Real per-qubit readout error alone varies by nearly 3x across the six qubits used (1.5% to 4.4%), which a uniform noise model cannot represent at all.

![Figure 5. Real per-qubit calibration data (readout error, T1) from the frozen ibmq_mumbai snapshot.](figures/fig6_real_calibration_data.png)

*Figure 5. Real per-qubit calibration data (readout error, T1) from the frozen ibmq_mumbai snapshot used to build the noise model, for the 6 physical qubits used in this experiment.*

| Metric | Raw noisy | ZNE | ML-corrected |
|---|---|---|---|
| MSE (random split) | 0.00180 | 0.00071 | 0.00046 |
| MAE (random split) | 0.0354 | 0.0221 | 0.0162 |
| MSE (OOD split, 6-qubit) | 0.00171 | 0.00075 | 0.00043 |
| MAE (OOD split, 6-qubit) | 0.0347 | 0.0228 | 0.0166 |

![Figure 6. MSE comparison side-by-side under the invented uniform-noise model and the real ibmq_mumbai-calibrated noise model.](figures/fig5_invented_vs_real_mse.png)

*Figure 6. MSE comparison side-by-side under the invented uniform-noise model (left, Section 3.1) and the real ibmq_mumbai-calibrated noise model (right). The ML-vs-ZNE-vs-raw ordering is unchanged; absolute error levels differ because the real device's specific error rates differ from the invented constants.*

Under the real-calibration noise model, the ML-corrected estimate reduces MSE by **74.3%** relative to raw noisy output and **35.1%** relative to ZNE on the random split, and **74.6%/42.1%** respectively on the out-of-distribution (unseen 6-qubit) split. Both the direction and the rough magnitude of the result are consistent with the invented-noise-model experiment in Section 3.1 — the ML correction's advantage over ZNE is not an artifact of a convenient, hand-picked noise model.

Feature importances shift under the real model (the ZNE point estimate itself becomes the dominant input, at 0.81 importance, versus a more even split between raw noisy, ZNE, and folded measurements under the invented model) — a reminder that the specific way the ML model uses its inputs is noise-model-dependent even though the qualitative conclusion (ML beats ZNE) is not.

**What this does and does not establish:** this is still a classical simulation, not an execution on live quantum hardware — `FakeMumbaiV2` is a frozen snapshot IBM ships specifically for reproducible offline testing, and calibration drifts over time on a real device in ways this snapshot cannot capture. What it does establish is that the core result is not an artifact of the arbitrary uniform noise constants used in Section 3.1: it holds when every error rate in the simulation is a real number measured on a real 27-qubit processor, with realistic per-qubit heterogeneity, real T1/T2 decoherence, and zero invented constants.

---

## 4. Discussion

The result that matters most here is not the in-distribution number — it is that the improvement survives the out-of-distribution test. A model that only interpolated within its training circuit sizes would be expected to degrade sharply when asked to correct a 6-qubit circuit it had never structurally seen. Instead it retained roughly 80% of its relative advantage over ZNE (39.9% vs 49.0% improvement). This suggests the Gradient Boosting model has learned something closer to a general noisy-to-ideal mapping conditioned on the folded measurements, similar in spirit to what vnCDR aims for with a linear ansatz.

It is also worth being explicit about what this project does not show. It does not demonstrate an improvement on real quantum hardware, where noise is time-varying, spatially inhomogeneous across qubits, and only partially captured by any static depolarizing model (see Section 6). It also does not compare against Probabilistic Error Cancellation (PEC) or Clifford Data Regression directly, both of which are established learning-adjacent or exact QEM baselines with different resource trade-offs than ZNE.

---

## 5. Reproducibility

All code, the generated dataset (480 circuits), trained model artifacts, and figures are included alongside this report. The pipeline is fully deterministic given the fixed random seeds used for circuit generation and simulation.

- `circuits.py` — benchmark circuit generation and exact statevector ground truth
- `noise_sim.py` — Aer noise model, gate folding, and ZNE extrapolation
- `generate_dataset.py` — builds the 480-circuit dataset (`dataset.csv`)
- `train_eval.py` — trains the Gradient Boosting model and evaluates both regimes
- `make_figures.py` — generates all figures in this report
- `real_noise.py` — builds the real-hardware-calibrated noise model from the `ibmq_mumbai` snapshot
- `generate_dataset_real.py` — builds the 480-circuit dataset under real-calibration noise (`dataset_real_noise.csv`)
- `train_eval_real.py`, `make_comparison_figures.py` — real-noise-model training/evaluation and comparison figures

---

## 6. Limitations and Threats to Validity

- The real-calibration experiment (Section 3.4) still uses a frozen, historical snapshot (`FakeMumbaiV2`), not a live pull from a currently-running device. Real hardware calibration drifts day to day; a model trained on one snapshot may not transfer to another, or to the same device recalibrated (a known limitation of learning-based QEM discussed explicitly in the CDR/vnCDR literature).
- No circuits were executed on live quantum hardware in either experiment. Queue access and cost make this impractical within the project's timeframe; all results are from Qiskit Aer noisy simulation, using either an invented uniform noise model or a real device's frozen calibration snapshot, not a live IBM Quantum backend.
- Single observable, single circuit family: only ⟨Z₀⟩ on a hardware-efficient-ansatz circuit family was tested. Generalization to other observables (e.g. multi-qubit correlators) or circuit families (e.g. QAOA cost Hamiltonians on structured graphs) was not evaluated.
- Shot noise: each expectation value is estimated from 3,000 shots, which itself introduces statistical noise into both the training labels' noisy-side inputs and the ZNE baseline, on top of the physical noise being modeled.
- No comparison to CDR/vnCDR directly, or to Probabilistic Error Cancellation, both established QEM baselines with different assumptions and resource costs than ZNE.

---

## 7. Conclusion

A Gradient Boosted Trees model trained on noisy and noise-scaled measurements, plus circuit-structure features, predicts ideal expectation values substantially more accurately than both raw noisy output and standard Zero-Noise Extrapolation, on 480 simulated hardware-efficient-ansatz circuits under a realistic depolarizing-plus-readout noise model. The improvement persists on circuit sizes withheld from training, suggesting genuine generalization rather than memorization. The clearest next steps are validating on real quantum hardware, testing additional circuit families and observables, and comparing directly against Clifford Data Regression and its noise-scaled variant.

---

## References

1. K. Temme, S. Bravyi, and J. M. Gambetta, "Error mitigation for short-depth quantum circuits," *Physical Review Letters*, vol. 119, 180509, 2017.
2. A. Kandala, K. Temme, A. D. Córcoles, A. Mezzacapo, J. M. Chow, and J. M. Gambetta, "Error mitigation extends the computational reach of a noisy quantum processor," *Nature*, vol. 567, pp. 491-495, 2019.
3. T. Giurgica-Tiron, Y. Hindy, R. LaRose, A. Mari, and W. J. Zeng, "Digital zero noise extrapolation for quantum error mitigation," in *Proc. IEEE Int. Conf. on Quantum Computing and Engineering (QCE)*, 2020, pp. 306-316.
4. S. Endo, S. C. Benjamin, and Y. Li, "Practical quantum error mitigation for near-future applications," *Physical Review X*, vol. 8, 031027, 2018.
5. P. Czarnik, A. Arrasmith, P. J. Coles, and L. Cincio, "Error mitigation with Clifford quantum-circuit data," *Quantum*, vol. 5, p. 592, 2021.
6. A. Lowe, M. H. Gordon, P. Czarnik, A. Arrasmith, P. J. Coles, and L. Cincio, "Unified approach to data-driven quantum error mitigation," *Physical Review Research*, vol. 3, 033098, 2021.
7. A. Strikis, D. Qin, Y. Chen, S. C. Benjamin, and Y. Li, "Learning-based quantum error mitigation," *PRX Quantum*, vol. 2, 040330, 2021.
8. Z. Cai, R. Babbush, S. C. Benjamin, S. Endo, W. J. Huggins, Y. Li, J. R. McClean, and T. E. O'Brien, "Quantum error mitigation," *Reviews of Modern Physics*, vol. 95, 045005, 2023.
9. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.
10. Qiskit contributors, "Qiskit: An open-source framework for quantum computing," and Qiskit Aer noise simulation framework, IBM Quantum.
11. Qiskit IBM Runtime fake-provider module (`qiskit_ibm_runtime.fake_provider`), providing frozen historical calibration snapshots of real IBM Quantum devices (including `FakeMumbaiV2` / `ibmq_mumbai`) for reproducible offline noisy simulation.
