# Quantum vs. Classical SVM Benchmark Experiment

**Author:** Josiah Chan (K23091949)  
**Institution:** King's College London  

## Overview
This repository contains the complete experimental framework for evaluating a **Quantum Support Vector Machine (QSVC)** against a **Classical SVM (RBF Kernel)** under controlled and realistic conditions.

The study adopts a systematic approach consisting of:

- Evaluation on standard benchmark datasets (e.g., MNIST, Fashion-MNIST, Breast Cancer)
- Comparison under idealised (statevector) and realistic (sampling-based) quantum settings
- Qualitative analysis of dataset structure
- Hypothesis-driven experimentation using targeted synthetic datasets

The framework is designed to isolate the effect of **feature representation**, ensuring a fair comparison between classical and quantum models within the same SVM optimisation pipeline.

The project utilises **Qiskit Machine Learning** primitives (`FidelityQuantumKernel`) and is configured to run on local simulators using Qiskit Aer's sampling backends, reflecting constraints of near-term quantum computing.

## Architecture and File Structure

* **`main.py`**: The interactive execution script. Run this to select your experiment mode via a terminal UI.
* **`experiment.py`**: Contains the `ExperimentRunner`. Orchestrates the experiment lifecycle: dataset initialisation, hyperparameter tuning, trial execution, statistical analysis, and Matplotlib visualisation.
* **`quantum_infrastructure.py`**: Contains the `QuantumProvider` class. Manages the Qiskit runtime environment, configures the local `AerSampler`, and builds the fidelity-based kernel evaluation functions.
* **`data_manager.py`**: Handles all data generation, PCA dimensionality reduction, and test/train splits (including Monte Carlo sub-sampling and K-Fold cross-validation).
* **`feature_map_factory.py`**: Generates the quantum data encoding circuits (e.g., `ZZFeatureMap`).
* **`tuner.py`**: Optimises model hyperparameters (C, gamma, circuit reps, entanglement) using grid search.

---

## Installation

This project requires a Python environment (3.9+ recommended) with Qiskit, Scikit-Learn, and their respective dependencies.

```bash
pip install -r requirements.txt
