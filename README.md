# Quantum vs. Classical SVM Benchmark Experiment

**Author:** Josiah Chan (K23091949)  
**Institution:** King's College London  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19651062.svg)](https://doi.org/10.5281/zenodo.19651062)

## Overview
This repository contains the complete experimental framework for evaluating a **Quantum Support Vector Machine (QSVC)** against a **Classical SVM (RBF Kernel)** under controlled and realistic conditions.

The study adopts a systematic approach consisting of:

- Evaluation on standard benchmark datasets (e.g., MNIST, Fashion-MNIST, Breast Cancer)
- Comparison under idealised (statevector) and realistic (sampling-based) quantum settings
- Qualitative analysis of dataset structure
- Hypothesis-driven experimentation using targeted synthetic datasets

The framework is designed to isolate the effect of **feature representation**, ensuring a fair comparison between classical and quantum models within the same SVM optimisation pipeline.

---

## Data Acquisition
The experimental results, raw trial logs, and benchmark datasets required for full reproduction are hosted on Zenodo. 

1. **Download the Dataset:** Obtain the `datasets.zip` file from:  
   https://doi.org/10.5281/zenodo.19651062
2. **Extraction:** Unzip the contents into the root directory of this project.
3. **Directory Verification:** Your local environment should be structured as follows for the scripts to resolve paths correctly:

```text
.
├── src/                # Python source files
│   ├── main.py
│   ├── experiment.py
│   └── ... 
├── datasets/               # <--- EXTRACTED FROM ZENODO
│   ├── mnist_train.csv     # Original benchmark datasets
│   └── ...        
├── requirements.txt
└── README.md
```
---

## Architecture and File Structure

* **`main.py`**: The interactive execution script. Run this to select your experiment mode via a terminal UI.
* **`experiment.py`**: Contains the `ExperimentRunner`. Orchestrates the experiment lifecycle: dataset initialisation, hyperparameter tuning, trial execution, statistical analysis, and Matplotlib visualisation.
* **`quantum_infrastructure.py`**: Contains the `QuantumProvider` class. Manages the Qiskit runtime environment, configures the local `AerSampler`, and builds the fidelity-based kernel evaluation functions.
* **`data_manager.py`**: Handles all data generation, PCA dimensionality reduction, and test/train splits (including Monte Carlo sub-sampling and K-Fold cross-validation).
* **`feature_map_factory.py`**: Generates the quantum data encoding circuits (e.g., `ZZFeatureMap`).
* **`tuner.py`**: Optimises model hyperparameters (C, gamma, circuit reps, entanglement) using grid search.

---

## Installation

### Prerequisites
- Python 3.12
- Access to a terminal/command prompt

### Setup
1. **Clone the repository:**
```bash
git clone https://github.com/josiahwkc/qml-dissertation.git
cd qml-dissertation
```
2. **Install dependencies:**
It is recommended to use a virtual environment to manage Qiskit versions:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage
To execute the experimental framework, run:
```bash
python src/main.py
```
Upon execution, follow the terminal prompts to select an experiment mode. The system will retrieve configurations from the `ExperimentConfig` class and execute the selected benchmark, saving all outputs to the `./results` directory
