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
from feature_map_factory import FeatureMapFactory


# Configs
NUM_DIMS = 5          # Number of PCA dimensions (and Qubits)
NUM_TRIALS = 10 #30        # Number of random seeds to average over (min 5 for stats)
N_CLASS = 2           # Binary classification
FIXED_SIZE = 100      # Default training size for non-'size' experiments

print("Choose an experiment mode:")
print("['size', 'imbalance', 'feature_complexity', 'margin', 'clusters', or 'noise']\n")
EXPERIMENT_MODE = input()


def main():
    provider = QuantumProvider()
    runner = ExperimentRunner(quantum_provider=provider)
    
    print(f"Initialising Experiment Runner in '{EXPERIMENT_MODE}' mode...")

    if EXPERIMENT_MODE == 'size':
        runner.initialise_datasets(
            num_dims=NUM_DIMS, 
            experiment_values=[20, 40, 60, 80, 100],
            mode='size',
            filename='fashion-mnist_train.csv',
            target_col='label'
        )
        runner.run_experiment(
            mode='size',num_dims=NUM_DIMS, num_trials=NUM_TRIALS, 
            experiment_values=[20, 40, 60, 80, 100]
        )
        
    elif EXPERIMENT_MODE == 'imbalance':
        runner.initialise_datasets(
            num_dims=NUM_DIMS,
            experiment_values=[0.5, 0.6, 0.7, 0.8, 0.9],
            mode='imbalance',
            filename='breast-cancer.csv',
            target_col='diagnosis'
        )
        runner.run_experiment(
            mode='imbalance', num_dims=NUM_DIMS, num_trials=NUM_TRIALS, 
            experiment_values=[0.5, 0.6, 0.7, 0.8, 0.9], fixed_size=FIXED_SIZE
        )
        
    elif EXPERIMENT_MODE == 'feature_complexity':
        runner.initialise_datasets(
            num_dims=NUM_DIMS,
            experiment_values=[1, 2, 3, 4, 5],
            mode='feature_complexity',
        )
        runner.run_experiment(
            mode='feature_complexity', num_dims=NUM_DIMS, num_trials=NUM_TRIALS, 
            experiment_values=[1, 2, 3, 4, 5], fixed_size=FIXED_SIZE
        )
        
    elif EXPERIMENT_MODE == 'margin':
        runner.initialise_datasets(
            num_dims=NUM_DIMS,
            experiment_values=[0.1, 0.5, 1.0, 1.5, 2.0],
            mode='margin',
        )
        runner.run_experiment(
            mode='margin', num_dims=NUM_DIMS, num_trials=NUM_TRIALS, 
            experiment_values=[0.1, 0.5, 1.0, 1.5, 2.0], fixed_size=FIXED_SIZE
        )
        
    elif EXPERIMENT_MODE == 'clusters':
        runner.initialise_datasets(
            num_dims=NUM_DIMS,
            experiment_values=[1, 2, 3, 4],
            mode='clusters',
        )
        runner.run_experiment(
            mode='clusters', num_dims=NUM_DIMS, num_trials=NUM_TRIALS, 
            experiment_values=[1, 2, 3, 4], fixed_size=FIXED_SIZE
        )
        
    elif EXPERIMENT_MODE == 'noise':
        runner.initialise_datasets(
            num_dims=NUM_DIMS,
            experiment_values=[0.0, 0.05, 0.10, 0.15, 0.20],
            mode='noise',
        )
        runner.run_experiment(
            mode='noise', num_dims=NUM_DIMS, num_trials=NUM_TRIALS, 
            experiment_values=[0.0, 0.05, 0.10, 0.15, 0.20], fixed_size=FIXED_SIZE
        )
     
if __name__ == "__main__":
    main()
# %%
