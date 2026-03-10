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
from sklearn.svm import SVC
from sklearn import metrics

from data_manager import CSVDataManager, AdhocDataManager, SyntheticDataManager
from classical_tuner import ClassicalSVMTuner

# Quantum Imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# %%
# Experiment Class
def nadeau_bengio_corrected_ttest(q_scores, c_scores, n_train, n_test):
    """
    Computes the Nadeau-Bengio corrected paired t-test for repeated 
    train/test splits. Corrects for the variance underestimation caused 
    by overlapping training sets.
    
    Ref: Nadeau, C., and Bengio, Y. (2003). Inference for the Generalization Error.
    """
    differences = np.array(q_scores) - np.array(c_scores)
    k = len(differences)
    
    if k < 2:
        return 1.0 # Need at least 2 trials for variance
        
    mean_diff = np.mean(differences)
    var_diff = np.var(differences, ddof=1)
    
    if var_diff == 0:
        return 1.0 # No variance, identical models
        
    # The Nadeau-Bengio variance correction
    # (1/k) accounts for the number of trials
    # (n_test/n_train) accounts for the overlap in training data
    corrected_variance = var_diff * ((1 / k) + (n_test / n_train))
    
    t_stat = mean_diff / np.sqrt(corrected_variance)
    
    # Calculate two-tailed p-value
    p_val = stats.t.sf(np.abs(t_stat), df=k-1) * 2
    return p_val

class ExperimentRunner():
    def __init__(self):
        self.results = {
            'x_values': [],
            'sizes': [],
            'q_acc': [], 'q_acc_std': [],  'q_f1': [], 'q_f1_std': [], 'q_time': [],
            'c_acc': [], 'c_acc_std': [], 'c_f1': [], 'c_f1_std': [], 'c_time': [],
            'delta_acc': [], 'p_val_acc': [], 'delta_f1': [], 'p_val_f1': []
        }
        
    def run_classical(self, X_train, X_test, y_train, y_test, cache_key=None):
        """
        Runs Classical SVM (RBF Kernel) with Hyperparameter Tuning
        """
        best_params = ClassicalSVMTuner.get_best_params(X_train, y_train, cache_key)
        clf = SVC(kernel='rbf', **best_params)
    
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
        
        start_time = time.time()
        
        try:
            matrix_train = kernel.evaluate(x_vec=X_train)
            matrix_test = kernel.evaluate(x_vec=X_test, y_vec=X_train)
        except ValueError as e:
            print(f"\nCRITICAL ERROR: {e}")
            exit()
            
        qsvm = SVC(kernel='precomputed')
        qsvm.fit(matrix_train, y_train)
        
        end_time = time.time()
        
        y_pred = qsvm.predict(matrix_test)
        
        # Calculate metrics
        score = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, average='weighted')
        
        return score, f1, (end_time - start_time)
    
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
            x_label = 'Number of Informative Features'
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
            raise ValueError(f"Invalid mode: {mode}")
        
        # PRE-GENERATE datasets for synthetic modes
        datasets_dict = {}
        if mode in ['feature_complexity', 'margin', 'clusters', 'noise']:
            for value in experiment_values:
                if mode == 'feature_complexity':
                    datasets_dict[value] = SyntheticDataManager(
                        num_dims=num_dims, 
                        n_informative=value,
                    )
                elif mode == 'margin':
                    datasets_dict[value] = SyntheticDataManager(
                        num_dims=num_dims, 
                        class_sep=value,
                    )
                elif mode == 'clusters':
                    datasets_dict[value] = SyntheticDataManager(
                        num_dims=num_dims, 
                        n_clusters_per_class=value,
                    )
                elif mode == 'noise':
                    datasets_dict[value] = SyntheticDataManager(
                        num_dims=num_dims, 
                        flip_y=value,
                    )
        
        for value in experiment_values:
            c_data = {'acc': [], 'f1': [], 'time': []}
            q_data = {'acc': [], 'f1': [], 'time': []}
            
            print(f"\n{'='*60}")
            print(f"RUNNING EXPERIMENT: {value_name} = {value}")
            print(f"{'='*60}")
            
            if mode in ['feature_complexity', 'margin', 'clusters', 'noise']:
                data_manager = datasets_dict[value]
            
            for trial in range(num_trials):
                
                # Get data based on mode
                if mode == 'size':
                    X_tr, X_te, y_tr, y_te = data_manager.get_data_split(
                        train_size=value, seed=trial, imbalance_ratio=0.5
                    )
                elif mode == 'imbalance':
                    X_tr, X_te, y_tr, y_te = data_manager.get_data_split(
                        train_size=fixed_size, seed=trial, imbalance_ratio=value
                    )
                else:
                    X_tr, X_te, y_tr, y_te = data_manager.get_data_split(
                        train_size=fixed_size, seed=trial, imbalance_ratio=0.5
                    )
                    
                print(f"\nTrial {trial+1}/{num_trials}...")
                
                # Run Classical
                cache_key = f"{mode}_{value}"
                c_score, c_f1, c_time = self.run_classical(X_tr, X_te, y_tr, y_te, cache_key=cache_key)
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
            
            # Effect Size (Cohen's d)
            # Pooled standard deviation calculation
            std_pooled_acc = np.sqrt((q_std_acc**2 + c_std_acc**2) / 2)
            std_pooled_f1 = np.sqrt((q_std_f1**2 + c_std_f1**2) / 2)
            
            # Protect against division by zero if all trials are perfectly identical
            if std_pooled_acc == 0:
                cohen_d_acc = float('inf') if abs(delta_acc) > 0 else 0.0
            else:
                cohen_d_acc = delta_acc / std_pooled_acc
                
            if std_pooled_f1 == 0:
                cohen_d_f1 = float('inf') if abs(delta_f1) > 0 else 0.0
            else:
                cohen_d_f1 = delta_f1 / std_pooled_f1
                            
            # NADEAU-BENGIO CORRECTED T-TEST
            # Extract current train/test sizes from the last trial loop
            n_train = len(X_tr)
            n_test = len(X_te)
            
            p_val_acc = nadeau_bengio_corrected_ttest(q_data['acc'], c_data['acc'], n_train, n_test)
            p_val_f1 = nadeau_bengio_corrected_ttest(q_data['f1'], c_data['f1'], n_train, n_test)
            
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
            print("\n[ STATISTICAL ANALYSIS (Nadeau-Bengio Corrected t-test & Effect Size) ]")
            print("-" * 85)
            print(f'{"Metric":<8} | {"Delta (Q-C)":<12} | {"Cohen\'s d":<10} | {"p-value":<10} | {"Significant (p<0.05)?"}')
            print("-" * 85)
            sig_acc = "YES (*)" if p_val_acc < 0.05 else "NO"
            sig_f1 = "YES (*)" if p_val_f1 < 0.05 else "NO"
            
            print(f"{'Accuracy':<8} | {delta_acc:>+12.2%} | {cohen_d_acc:>10.2f} | {p_val_acc:>10.4f} | {sig_acc}")
            print(f"{'F1 Score':<8} | {delta_f1:>+12.2f} | {cohen_d_f1:>10.2f} | {p_val_f1:>10.4f} | {sig_f1}")
            print(f"{'Time':<8} | QSVM was {time_ratio:.0f}x slower")
            print("-" * 85)
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
