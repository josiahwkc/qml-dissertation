"""
Quantum vs. Classical SVM Benchmark Experiment
==============================================
Author: Josiah Chan (K23091949)
Date: February 2026
Description: 
  This script benchmarks the performance and training time of a Quantum Support 
  Vector Machine (QSVC) against a Classical SVM (RBF Kernel).
  
  Key Features:
  - Data Diet Analysis: Testing performance across varying training set sizes.
  - Dimensionality Reduction: PCA pipeline to map 64-pixel images to N-qubits.

Attribution:
  - Base QSVC logic adapted from IBM Qiskit Summer School 2021 (Lab 3).
  - Original URL: https://github.com/Qiskit/platypus/blob/main/notebooks/summer-school/2021/resources/lab-notebooks/lab-3.ipynb
"""

# %%
# Imports
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Quantum Imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# %%
# Data Manager Class
class DataManager():
    def __init__(self, num_dims=4, n_class=2):
        self.num_dims = num_dims
        digits = datasets.load_digits(n_class=n_class)
        self.X = digits.data
        self.y = digits.target
        
    def get_data_split(self, train_size, seed, imbalance_ratio=0.5):
        """
        imbalance_ratio: proportion of class 0 in training set
        0.5 = balanced, 0.9 = 90% class 0, 10% class 1
        """
        # Separate classes
        X_class0 = self.X[self.y == 0]
        X_class1 = self.X[self.y == 1]
        
        # Calculate how many of each class to sample
        n_class0 = int(train_size * imbalance_ratio)
        n_class1 = train_size - n_class0
        
        rng = np.random.default_rng(seed)
        idx0 = rng.choice(len(X_class0), size=n_class0, replace=False)
        idx1 = rng.choice(len(X_class1), size=n_class1, replace=False)
        
        X_train = np.vstack([X_class0[idx0], X_class1[idx1]])
        y_train = np.array([0] * n_class0 + [1] * n_class1)
        
        # Shuffle
        shuffle_idx = rng.permutation(len(y_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        
        _, X_test, _, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=seed, stratify=self.y
        )
        
        preprocessing_pipeline = Pipeline([
                ('pca', PCA(n_components=self.num_dims)),
                ('std_scaler', StandardScaler()),
                ('minmax_scaler', MinMaxScaler(feature_range=(-1, 1)))
            ])

        X_train_processed = preprocessing_pipeline.fit_transform(X_train)
        X_test_processed = preprocessing_pipeline.transform(X_test)
        
        return X_train_processed, X_test_processed, y_train, y_test

# %%
# Experiment Class
class ExperimentRunner():
    def __init__(self):
        self.results = {
            'x_values': [],
            'sizes': [],
            'q_acc': [], 'q_acc_std': [],  'q_f1': [], 'q_f1_std': [], 'q_time': [],
            'c_acc': [], 'c_acc_std': [], 'c_f1': [], 'c_f1_std': [], 'c_time': []
        }
        
    def run_classical(self, X_train, X_test, y_train, y_test):
        """
        Runs Classical SVM (RBF Kernel)
        """
        clf = svm.SVC(kernel='rbf')
        
        start_time = time.time()
        clf.fit(X_train, y_train)
        end_time = time.time()
        
        y_pred = clf.predict(X_test)
        
        score = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        
        return score, f1, (end_time - start_time)
    
    def run_quantum(self, X_train, X_test, y_train, y_test, num_dims):
        """
        Runs Quantum SVM (ZZFeatureMap + FidelityKernel)
        """
        feature_map = ZZFeatureMap(feature_dimension=num_dims, reps=2, entanglement='linear')
        kernel = FidelityQuantumKernel(feature_map=feature_map)
        
        start = time.time()
        
        try:
            matrix_train = kernel.evaluate(x_vec=X_train)
            matrix_test = kernel.evaluate(x_vec=X_test, y_vec=X_train)
        except ValueError as e:
            print(f"\nCRITICAL ERROR: {e}")
            print("This usually means N_DIM is too high or PCA failed.")
            exit()
            
        qsvm = svm.SVC(kernel='precomputed')
        qsvm.fit(matrix_train, y_train)
        end = time.time()
        
        y_pred = qsvm.predict(matrix_test)
        
        score = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred)
        
        return score, f1, (end - start)
    
    def run_experiment(self, mode, data_manager, num_dims, num_trials, 
                    training_sizes=None, imbalance_ratios=None, fixed_size=100):
        """
        Main experiment runner
        
        Args:
            mode: 'size' or 'imbalance'
            data_manager: DataManager instance
            num_dims: Number of PCA dimensions
            num_trials: Number of trials per configuration
            training_sizes: List of sizes (required if mode='size')
            imbalance_ratios: List of ratios (required if mode='imbalance')
            fixed_size: Training size when mode='imbalance'
        """
        if mode == 'size':
            if training_sizes is None:
                raise ValueError("training_sizes must be provided when mode='size'")
            experiment_values = training_sizes
            x_label = 'Training Samples'
            title_suffix = 'vs Training Size'
            value_name = 'Size'
        elif mode == 'imbalance':
            if imbalance_ratios is None:
                raise ValueError("imbalance_ratios must be provided when mode='imbalance'")
            experiment_values = imbalance_ratios
            x_label = 'Class Imbalance Ratio (proportion of class 0)'
            title_suffix = 'vs Class Imbalance'
            value_name = 'Ratio'
        else:
            raise ValueError("mode must be 'size' or 'imbalance'")
        
        for value in experiment_values:
            c_data = {'acc': [], 'f1': [], 'time': []}
            q_data = {'acc': [], 'f1': [], 'time': []}
            
            print(f"\n{'='*60}")
            print(f"RUNNING EXPERIMENT: {value_name} = {value}")
            print(f"{'='*60}")
            
            for seed in range(num_trials):
                # Get data based on mode
                if mode == 'size':
                    X_tr, X_te, y_tr, y_te = data_manager.get_data_split(
                        train_size=value, seed=seed, imbalance_ratio=0.5
                    )
                else:  # mode == 'imbalance'
                    X_tr, X_te, y_tr, y_te = data_manager.get_data_split(
                        train_size=fixed_size, seed=seed, imbalance_ratio=value
                    )
                
                print(f"\nTrial {seed+1}/{num_trials}...")
                
                # Run Classical
                c_score, c_f1, c_time = self.run_classical(X_tr, X_te, y_tr, y_te)
                c_data['acc'].append(c_score)
                c_data['f1'].append(c_f1)
                c_data['time'].append(c_time)
                
                # Run Quantum
                q_score, q_f1, q_time = self.run_quantum(X_tr, X_te, y_tr, y_te, num_dims)
                q_data['acc'].append(q_score)
                q_data['f1'].append(q_f1)
                q_data['time'].append(q_time)
            
            # Calculate statistics
            q_avg_acc = np.mean(q_data['acc'])
            q_std_acc = np.std(q_data['acc'])
            q_avg_f1 = np.mean(q_data['f1'])
            q_std_f1 = np.std(q_data['f1'])
            q_avg_time = np.mean(q_data['time'])
            
            c_avg_acc = np.mean(c_data['acc'])
            c_std_acc = np.std(c_data['acc'])
            c_avg_f1 = np.mean(c_data['f1'])
            c_std_f1 = np.std(c_data['f1'])
            c_avg_time = np.mean(c_data['time'])
            
            # --- PRINT TABLE 1: QUANTUM ---
            print("\n[ QUANTUM MODEL PERFORMANCE ]")
            print("-" * 50)
            print(f"{'Trial':<6} | {'Acc':<10} | {'F1':<10} | {'Time (s)':<10}")
            print("-" * 50)
            for i in range(num_trials):
                print(f"{i+1:<6} | {q_data['acc'][i]:<10.2%} | {q_data['f1'][i]:<10.2f} | {q_data['time'][i]:<10.4f}")
            print("-" * 50)
            print(f"{'AVG':<6} | {q_avg_acc:<10.2%} | {q_avg_f1:<10.2f} | {q_avg_time:<10.4f}")
            print(f"{'STD':<6} | {q_std_acc:<10.2%} | {q_std_f1:<10.2f} | {'-':<10}")
            print("-" * 50)

            # --- PRINT TABLE 2: CLASSICAL ---
            print("\n[ CLASSICAL MODEL PERFORMANCE ]")
            print("-" * 50)
            print(f"{'Trial':<6} | {'Acc':<10} | {'F1':<10} | {'Time (s)':<10}")
            print("-" * 50)
            for i in range(num_trials):
                print(f"{i+1:<6} | {c_data['acc'][i]:<10.2%} | {c_data['f1'][i]:<10.2f} | {c_data['time'][i]:<10.4f}")
            print("-" * 50)
            print(f"{'AVG':<6} | {c_avg_acc:<10.2%} | {c_avg_f1:<10.2f} | {c_avg_time:<10.4f}")
            print(f"{'STD':<6} | {c_std_acc:<10.2%} | {c_std_f1:<10.2f} | {'-':<10}")
            print("-" * 50)
            print("\n")
            
            # Store results
            self.results['x_values'].append(value)
            self.results['c_acc'].append(c_avg_acc)
            self.results['c_acc_std'].append(c_std_acc)
            self.results['c_f1'].append(c_avg_f1)
            self.results['c_f1_std'].append(c_std_f1)
            self.results['c_time'].append(c_avg_time)
            
            self.results['q_acc'].append(q_avg_acc)
            self.results['q_acc_std'].append(q_std_acc)
            self.results['q_f1'].append(q_avg_f1)
            self.results['q_f1_std'].append(q_std_f1)
            self.results['q_time'].append(q_avg_time)
        
        # Plot results after all values complete
        self.plot_results(x_label, title_suffix)
    
    def plot_results(self, x_label, title_suffix):
        """Generate comparison plots"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Accuracy with error bars
        ax1.errorbar(self.results['x_values'], self.results['c_acc'], 
                     yerr=self.results['c_acc_std'], fmt='o-', capsize=5, 
                     label='Classical', color='blue')
        ax1.errorbar(self.results['x_values'], self.results['q_acc'], 
                     yerr=self.results['q_acc_std'], fmt='s-', capsize=5, 
                     label='Quantum', color='purple')
        ax1.set_title(f'Accuracy {title_suffix}')
        ax1.set_xlabel(x_label)
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # F1 Score with error bars
        ax2.errorbar(self.results['x_values'], self.results['c_f1'], 
                     yerr=self.results['c_f1_std'], fmt='o-', capsize=5, 
                     label='Classical', color='blue')
        ax2.errorbar(self.results['x_values'], self.results['q_f1'], 
                     yerr=self.results['q_f1_std'], fmt='s-', capsize=5, 
                     label='Quantum', color='purple')
        ax2.set_title(f'F1 Score {title_suffix}')
        ax2.set_xlabel(x_label)
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        ax2.grid(True)
        
        # Time (log scale)
        ax3.plot(self.results['x_values'], self.results['c_time'], 
                 'o-', label='Classical', color='blue')
        ax3.plot(self.results['x_values'], self.results['q_time'], 
                 's-', label='Quantum', color='purple')
        ax3.set_title(f'Training Time {title_suffix}')
        ax3.set_yscale('log')
        ax3.set_xlabel(x_label)
        ax3.set_ylabel('Time (s)')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.show()

# %%
# CONFIGURATION AND EXECUTION

# Choose experiment mode: 'size' or 'imbalance'
EXPERIMENT_MODE = 'imbalance'

NUM_DIMS = 4
NUM_TRIALS = 3

data_manager = DataManager(num_dims=NUM_DIMS)
runner = ExperimentRunner()

if EXPERIMENT_MODE == 'size':
    # Training size experiment (balanced classes)
    TRAINING_SIZES = [20, 40, 60, 80, 100]
    runner.run_experiment(
        mode='size',
        data_manager=data_manager,
        num_dims=NUM_DIMS,
        num_trials=NUM_TRIALS,
        training_sizes=TRAINING_SIZES
    )
    
elif EXPERIMENT_MODE == 'imbalance':
    # Class imbalance experiment (fixed size)
    IMBALANCE_RATIOS = [0.5, 0.6, 0.7, 0.8, 0.9]
    FIXED_SIZE = 100
    runner.run_experiment(
        mode='imbalance',
        data_manager=data_manager,
        num_dims=NUM_DIMS,
        num_trials=NUM_TRIALS,
        imbalance_ratios=IMBALANCE_RATIOS,
        fixed_size=FIXED_SIZE
    )
# %%
