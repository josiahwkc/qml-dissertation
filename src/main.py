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

DATASETS = {
    'mnist_train.csv': 'label',
    'fashion-mnist_train.csv': 'label',
    'breast-cancer.csv': 'diagnosis'
}

CSV_MODES = [
    'size', 
    'imbalance', 
]

SYNTHETIC_MODES = [
    'feature_complexity', 
    'margin', 
    'clusters', 
    'noise', 
    'centroids_distance', 
    'cluster_spread',
    
]

QUANTUM_MODES = [
    'quantum_benchmark'
]

def get_user_choice(options):
    """Helper function to print a numbered menu and safely get user selection."""
    for i, option in enumerate(options, start=1):
        display_name = option.replace('_', ' ').title() if isinstance(option, str) else option
        print(f"[{i}] {display_name}")

    while True:
        try:
            choice = int(input("Enter number > ").strip())
            if 1 <= choice <= len(options):
                return options[choice - 1] # Return the actual string/value selected
            else:
                print(f"Error: Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Error: Invalid input. Please type a number.")

def main():
    print("Which type of data would you like to experiment on?") 
    data_type = get_user_choice(["CSV Data", "Synthetic Data", "Quantum Data"])
    
    filename = None
    target_col = None
    experiment_mode = None
    
    if data_type == "CSV Data":
        print("\nWhich dataset would you like to use?")
        # Convert dictionary keys into a list for the menu
        dataset_keys = list(DATASETS.keys())
        filename = get_user_choice(dataset_keys)
        target_col = DATASETS[filename]
        
        print("\nWhich variable do you want to experiment on?")
        experiment_mode = get_user_choice(CSV_MODES)
        
    elif data_type == "Synthetic Data":
        print("\nWhich variable do you want to experiment on?")
        experiment_mode = get_user_choice(SYNTHETIC_MODES)
    else:
        experiment_mode = "quantum_benchmark"
        
    provider = QuantumProvider()
    runner = ExperimentRunner(quantum_provider=provider)
    runner.run_experiment(mode=experiment_mode, filename=filename, target_col=target_col)
     
if __name__ == "__main__":
    main()
# %%
