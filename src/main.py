"""
Quantum vs. Classical SVM Benchmark - Main Execution Script
===========================================================
Author: Josiah Chan (K23091949)

Description:
  This is the main entry point for the benchmarking experiments. 
  Adjust the configuration variables below to run different tests.
"""

# %%
# EXECUTE
from experiment import ExperimentRunner
from quantum_infrastructure import QuantumProvider

# Configs
NUM_DIMS = 5          # Number of PCA dimensions (and Qubits)
NUM_TRIALS = 10 #30        # Number of random seeds to average over (min 5 for stats)
N_CLASS = 2           # Binary classification
FIXED_SIZE = 100      # Default training size for non-'size' experiments

SWEEP_VALUES_DICT = {
    'size': [20, 40, 60, 80, 100],
    'imbalance': [0.5, 0.6, 0.7, 0.8, 0.9],
    'feature_complexity': [1, 2, 3, 4, 5],
    'margin': [0.1, 0.5, 1.0, 1.5, 2.0],
    'custers': [1, 2, 3, 4],
    'noise': [0.0, 0.05, 0.10, 0.15, 0.20]
}


def main():
    print("Choose an experiment mode:")
    print("['size', 'imbalance', 'feature_complexity', 'margin', 'clusters', or 'noise']\n")
    experiment_mode = input()
    
    provider = QuantumProvider()
    runner = ExperimentRunner(
        quantum_provider=provider, 
        sweep_values_dict=SWEEP_VALUES_DICT, 
        num_dims=NUM_DIMS,
        num_trials=NUM_TRIALS,
        fixed_size=FIXED_SIZE
    )
    
    print(f"Initialising Experiment Runner in '{experiment_mode}' mode...")

    if experiment_mode == 'size':
        runner.initialise_datasets(mode='size', filename='fashion-mnist_train.csv', target_col='label')
        runner.run_experiment(mode='size')
        
    elif experiment_mode == 'imbalance':
        runner.initialise_datasets(mode='imbalance', filename='breast-cancer.csv', target_col='diagnosis')
        runner.run_experiment(mode='imbalance')
    
    # Perform experiment on synthetic datasets
    else:
        runner.initialise_datasets(mode=experiment_mode)
        runner.run_experiment(mode=experiment_mode)
     
if __name__ == "__main__":
    main()
# %%
