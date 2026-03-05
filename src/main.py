"""
Quantum vs. Classical SVM Benchmark - Main Execution Script
===========================================================
Author: Josiah Chan (K23091949)

Description:
  This is the main entry point for the benchmarking experiments. 
  Adjust the configuration variables below to run different tests.
"""

from experiment import ExperimentRunner
from data_manager import CSVDataManager, AdhocDataManager, SyntheticDataManager


# GLOBAL CONFIGURATION
NUM_DIMS = 4          # Number of PCA dimensions (and Qubits)
NUM_TRIALS = 5        # Number of random seeds to average over (min 5 for stats)
N_CLASS = 2           # Binary classification
FIXED_SIZE = 100       # Default training size for non-'size' experiments

# Choose experiment mode: 
# 'size', 'imbalance', 'feature_complexity', 'margin', 'clusters', or 'noise'
EXPERIMENT_MODE = 'feature_complexity'


def main():
    print(f"Initializing Experiment Runner in '{EXPERIMENT_MODE}' mode...")
    runner = ExperimentRunner()

    if EXPERIMENT_MODE == 'size':
        csv = CSVDataManager(filename="breast-cancer.csv", target_col="diagnosis", num_dims=NUM_DIMS, n_class=N_CLASS)
        runner.run_experiment(
            mode='size', data_manager=csv, num_dims=NUM_DIMS, num_trials=NUM_TRIALS, 
            experiment_values=[20, 40, 60, 80, 100]
        )
        
    elif EXPERIMENT_MODE == 'imbalance':
        csv = CSVDataManager(filename="breast-cancer.csv", target_col="diagnosis", num_dims=NUM_DIMS, n_class=N_CLASS)
        runner.run_experiment(
            mode='imbalance', data_manager=csv, num_dims=NUM_DIMS, num_trials=NUM_TRIALS, 
            experiment_values=[0.5, 0.6, 0.7, 0.8, 0.9], fixed_size=FIXED_SIZE
        )
        
    elif EXPERIMENT_MODE == 'feature_complexity':
        runner.run_experiment(
            mode='feature_complexity', data_manager=None, num_dims=NUM_DIMS, num_trials=NUM_TRIALS, 
            experiment_values=[2, 6, 10, 14, 18], fixed_size=FIXED_SIZE
        )
        
    elif EXPERIMENT_MODE == 'margin':
        runner.run_experiment(
            mode='margin', data_manager=None, num_dims=NUM_DIMS, num_trials=NUM_TRIALS, 
            experiment_values=[0.1, 0.5, 1.0, 1.5, 2.0], fixed_size=FIXED_SIZE
        )
        
    elif EXPERIMENT_MODE == 'clusters':
        runner.run_experiment(
            mode='clusters', data_manager=None, num_dims=NUM_DIMS, num_trials=NUM_TRIALS, 
            experiment_values=[1, 2, 3, 4], fixed_size=FIXED_SIZE
        )
        
    elif EXPERIMENT_MODE == 'noise':
        runner.run_experiment(
            mode='noise', data_manager=None, num_dims=NUM_DIMS, num_trials=NUM_TRIALS, 
            experiment_values=[0.0, 0.05, 0.10, 0.15, 0.20], fixed_size=FIXED_SIZE
        )

if __name__ == "__main__":
    main()