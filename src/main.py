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

def main():
    print("Choose an experiment mode:")
    print("['size', 'imbalance', 'feature_complexity', 'margin', 'clusters', 'noise', 'inter_distance', 'intra_spread, or 'quantum_benchmark']\n")
    experiment_mode = input()
    
    provider = QuantumProvider()
    runner = ExperimentRunner(quantum_provider=provider)
    
    print(f"Initialising Experiment Runner in '{experiment_mode}' mode...")

    if experiment_mode == 'size':
        runner.initialise_datasets(mode=experiment_mode, filename='mnist_train.csv', target_col='label')
        runner.run_experiment(mode=experiment_mode)
        
        runner.initialise_datasets(mode=experiment_mode, filename='fashion-mnist_train.csv', target_col='label')
        runner.run_experiment(mode=experiment_mode)
        
        runner.initialise_datasets(mode=experiment_mode, filename='breast-cancer.csv', target_col='diagnosis')
        runner.run_experiment(mode=experiment_mode)
        
    elif experiment_mode == 'imbalance':
        runner.initialise_datasets(mode=experiment_mode, filename='mnist_train.csv', target_col='label')
        runner.run_experiment(mode=experiment_mode)
        
        runner.initialise_datasets(mode=experiment_mode, filename='fashion-mnist_train.csv', target_col='label')
        runner.run_experiment(mode=experiment_mode)
        
        runner.initialise_datasets(mode=experiment_mode, filename='breast-cancer.csv', target_col='diagnosis')
        runner.run_experiment(mode=experiment_mode)
    
    # Perform experiment on synthetic datasets
    else:
        runner.initialise_datasets(mode=experiment_mode)
        runner.run_experiment(mode=experiment_mode)
     
if __name__ == "__main__":
    main()
# %%
