"""
Quantum vs. Classical SVM Benchmark Experiment
==============================================
Author: Josiah Chan (K23091949)

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
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import svm, metrics

from data_manager import CSVDataManager, AdhocDataManager, SyntheticDataManager

# Quantum Imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# %%
# Experiment Class
class ExperimentRunner():
    def __init__(self):
        self.results = {
            'x_values': [],
            'sizes': [],
            'q_acc': [], 'q_acc_std': [],  'q_f1': [], 'q_f1_std': [], 'q_time': [],
            'c_acc': [], 'c_acc_std': [], 'c_f1': [], 'c_f1_std': [], 'c_time': [],
            'delta_acc': [], 'p_val_acc': [], 'delta_f1': [], 'p_val_f1': []
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
        f1 = metrics.f1_score(y_test, y_pred, average='weighted')
        
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
        f1 = metrics.f1_score(y_test, y_pred, average='weighted')
        
        return score, f1, (end - start)
    
    def run_experiment(self, mode, data_manager, num_dims, num_trials, 
                    experiment_values=None, fixed_size=100):
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
        if experiment_values is None:
            raise ValueError(f"experiment_values must be provided for mode='{mode}'")
            
        if mode == 'size':
            x_label = 'Training Samples'
            title_suffix = 'vs Training Size'
            value_name = 'Training Size'
        elif mode == 'imbalance':
            x_label = 'Class Imbalance Ratio (proportion of class 0)'
            title_suffix = 'vs Class Imbalance'
            value_name = 'Ratio'
        elif mode == 'feature_complexity':
            x_label = 'Number of Informative Features (Before PCA)'
            title_suffix = 'vs Informative Features'
            value_name = 'Informative Features'
        elif mode == 'margin':
            x_label = 'Class Separation Margin (class_sep)'
            title_suffix = 'vs Class Separation'
            value_name = 'Class Separation'
        elif mode == 'clusters':
            x_label = 'Clusters per Class (Decision Boundary Non-Linearity)'
            title_suffix = 'vs Clusters per Class'
            value_name = 'Clusters/Class'
        elif mode == 'noise':
            x_label = 'Label Noise Fraction (flip_y)'
            title_suffix = 'vs Label Noise'
            value_name = 'Noise Fraction'
        else:
            raise ValueError("mode must be 'size', 'imbalance', 'feature_complexity', or 'margin'")
        
        for value in experiment_values:
            c_data = {'acc': [], 'f1': [], 'time': []}
            q_data = {'acc': [], 'f1': [], 'time': []}
            
            print(f"\n{'='*60}")
            print(f"RUNNING EXPERIMENT: {value_name} = {value}")
            print(f"{'='*60}")
            
            if mode == 'feature_complexity':
                print(f"Generating new synthetic dataset with n_informative={value}...")
                data_manager = SyntheticDataManager(num_dims=num_dims, n_informative=value)
            elif mode == 'margin':
                print(f"Generating new synthetic dataset with class_sep={value}...")
                data_manager = SyntheticDataManager(num_dims=num_dims, class_sep=value)
            elif mode == 'clusters':
                print(f"Generating new synthetic dataset with n_clusters_per_class={value}...")
                data_manager = SyntheticDataManager(num_dims=num_dims, n_clusters_per_class=value)
            elif mode == 'noise':
                print(f"Generating new synthetic dataset with flip_y={value}...")
                data_manager = SyntheticDataManager(num_dims=num_dims, flip_y=value)
            
            for seed in range(num_trials):
                # Get data based on mode
                if mode == 'size':
                    X_tr, X_te, y_tr, y_te = data_manager.get_data_split(
                        train_size=value, seed=seed, imbalance_ratio=0.5
                    )
                elif mode == 'imbalance':
                    X_tr, X_te, y_tr, y_te = data_manager.get_data_split(
                        train_size=fixed_size, seed=seed, imbalance_ratio=value
                    )
                else:
                    X_tr, X_te, y_tr, y_te = data_manager.get_data_split(
                        train_size=fixed_size, seed=seed, imbalance_ratio=0.5
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
            q_avg_acc, q_std_acc = np.mean(q_data['acc']), np.std(q_data['acc'])
            q_avg_f1, q_std_f1 = np.mean(q_data['f1']), np.std(q_data['f1'])
            q_avg_time = np.mean(q_data['time'])
            
            c_avg_acc, c_std_acc = np.mean(c_data['acc']), np.std(c_data['acc'])
            c_avg_f1, c_std_f1 = np.mean(c_data['f1']), np.std(c_data['f1'])
            c_avg_time = np.mean(c_data['time'])
            
            # Calculate Deltas (Quantum - Classical)
            delta_acc = q_avg_acc - c_avg_acc
            delta_f1 = q_avg_f1 - c_avg_f1
            time_ratio = q_avg_time / c_avg_time if c_avg_time > 0 else q_avg_time
            
            # Paired t-tests for Statistical Significance (p-value)
            # Catch warnings in case the arrays are identical (variance=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, p_val_acc = stats.ttest_rel(q_data['acc'], c_data['acc'])
                _, p_val_f1 = stats.ttest_rel(q_data['f1'], c_data['f1'])
                
            # If arrays are exactly the same, p-value returns NaN. Set to 1.0 (not significant).
            if np.isnan(p_val_acc): p_val_acc = 1.0
            if np.isnan(p_val_f1): p_val_f1 = 1.0
            
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
            
            # --- PRINT TABLE 3: STATISTICAL SIGNIFICANCE ---
            print("\n[ STATISTICAL ANALYSIS (Paired t-test) ]")
            print("-" * 65)
            print(f"{'Metric':<8} | {'Delta (Q - C)':<15} | {'p-value':<10} | {'Significant (p<0.05)?'}")
            print("-" * 65)
            sig_acc = "YES (*)" if p_val_acc < 0.05 else "NO"
            sig_f1 = "YES (*)" if p_val_f1 < 0.05 else "NO"
            print(f"{'Accuracy':<8} | {delta_acc:>+14.2%} | {p_val_acc:>10.4f} | {sig_acc}")
            print(f"{'F1 Score':<8} | {delta_f1:>+14.2f} | {p_val_f1:>10.4f} | {sig_f1}")
            print(f"{'Time':<8} | QSVM was {time_ratio:.0f}x slower")
            print("-" * 65)
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
            
            self.results['delta_acc'].append(delta_acc)
            self.results['p_val_acc'].append(p_val_acc)
            self.results['delta_f1'].append(delta_f1)
            self.results['p_val_f1'].append(p_val_f1)
        
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
NUM_DIMS = 4
NUM_TRIALS = 5
N_CLASS = 2
FIXED_SIZE = 100

# Choose experiment mode: 'size', 'imbalance', 'feature_complexity', 'margin', 'clusters', 'noise'
EXPERIMENT_MODE = 'clusters'


runner = ExperimentRunner()

if EXPERIMENT_MODE == 'size':
        csv = CSVDataManager(filename="breast-cancer.csv", target_col="diagnosis", num_dims=NUM_DIMS, n_class=N_CLASS)
        runner.run_experiment(mode='size', data_manager=csv, num_dims=NUM_DIMS, num_trials=NUM_TRIALS, experiment_values=[20, 40, 60, 80, 100])
        
elif EXPERIMENT_MODE == 'imbalance':
    csv = CSVDataManager(filename="breast-cancer.csv", target_col="diagnosis", num_dims=NUM_DIMS, n_class=N_CLASS)
    runner.run_experiment(mode='imbalance', data_manager=csv, num_dims=NUM_DIMS, num_trials=NUM_TRIALS, experiment_values=[0.5, 0.6, 0.7, 0.8, 0.9], fixed_size=FIXED_SIZE)
    
elif EXPERIMENT_MODE == 'feature_complexity':
    runner.run_experiment(mode='feature_complexity', data_manager=None, num_dims=NUM_DIMS, num_trials=NUM_TRIALS, experiment_values=[2, 6, 10, 14, 18], fixed_size=FIXED_SIZE)
    
elif EXPERIMENT_MODE == 'margin':
    runner.run_experiment(mode='margin', data_manager=None, num_dims=NUM_DIMS, num_trials=NUM_TRIALS, experiment_values=[0.1, 0.5, 1.0, 1.5, 2.0], fixed_size=FIXED_SIZE)
    
elif EXPERIMENT_MODE == 'clusters':
    # NOTE: n_informative must be >= n_classes * n_clusters_per_class
    runner.run_experiment(mode='clusters', data_manager=None, num_dims=NUM_DIMS, num_trials=NUM_TRIALS, experiment_values=[1, 2, 3, 4], fixed_size=FIXED_SIZE)
    
elif EXPERIMENT_MODE == 'noise':
    # 0.0 is perfect data, 0.20 means 20% of data has explicitly wrong labels
    runner.run_experiment(mode='noise', data_manager=None, num_dims=NUM_DIMS, num_trials=NUM_TRIALS, experiment_values=[0.0, 0.05, 0.10, 0.15, 0.20], fixed_size=FIXED_SIZE)
# %%
