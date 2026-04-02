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

from data_manager import CSVDataManager, AdhocDataManager, SyntheticDataManager, TrainingSampler
from tuner import ClassicalSVMTuner, QuantumSVMTuner

# Quantum Imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from feature_map_factory import FeatureMapFactory

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
    def __init__(self, quantum_provider):
        self.qp = quantum_provider
        self.classical_clf = SVC(kernel='rbf')
        self.reset()
      
    def reset(self):
        self.results = {
            'x_values': [],
            'sizes': [],
            'q_acc': [], 'q_acc_std': [], 'q_f1': [], 'q_f1_std': [], 'q_time': [],
            'c_acc': [], 'c_acc_std': [], 'c_f1': [], 'c_f1_std': [], 'c_time': [],
            'delta_acc': [], 'p_val_acc': [], 'delta_f1': [], 'p_val_f1': []
        }
       
    def run_classical(self, X_train, X_test, y_train, y_test, params):
        """Runs Classical SVM (RBF Kernel) with locked parameters"""        
        self.classical_clf.set_params(**params)
    
        start_time = time.time()
        self.classical_clf.fit(X_train, y_train)
        end_time = time.time()
        
        y_pred = self.classical_clf.predict(X_test)
        
        score = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, average='weighted')
        
        return score, f1, (end_time - start_time)
    
    def run_quantum(self, X_train, X_test, y_train, y_test, kernel, params):
        """Runs Quantum SVM with locked parameters"""        
        start_time = time.time()
        print("Quantum start")
        try:
            matrix_train = kernel.evaluate(x_vec=X_train)
            print("Train matrix done")
            matrix_test = kernel.evaluate(x_vec=X_test, y_vec=X_train)
            print("Test matrix done")
        except ValueError as e:
            print(f"\nCRITICAL ERROR: {e}")
            exit()
            
        qsvm = SVC(kernel='precomputed', C=params['C'])
        qsvm.fit(matrix_train, y_train)
        
        end_time = time.time()
        
        y_pred = qsvm.predict(matrix_test)
        
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
        
        # Resets and clears stored results
        self.reset()
           
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
                    datasets_dict[value] = SyntheticDataManager(num_dims=num_dims, n_informative=value)
                elif mode == 'margin':
                    datasets_dict[value] = SyntheticDataManager(num_dims=num_dims, class_sep=value)
                elif mode == 'clusters':
                    datasets_dict[value] = SyntheticDataManager(num_dims=num_dims, n_clusters_per_class=value)
                elif mode == 'noise':
                    datasets_dict[value] = SyntheticDataManager(num_dims=num_dims, flip_y=value)

        print("\n" + "="*80)
        print(" PHASE 1: HYPERPARAMETER OPTIMIZATION (HOLD-OUT VALIDATION)")
        print("="*80)
                
        if mode in ['feature_complexity', 'margin', 'clusters', 'noise']:
            # Use middle value as baseline (e.g., for [1,2,3,4,5], use value=3)
            baseline_value = experiment_values[len(experiment_values) // 2]
            tune_dm = datasets_dict[baseline_value]
            print(f"  Baseline dataset: {mode} = {baseline_value}")
        else:
            tune_dm = data_manager
            print(f"  Baseline dataset: {mode} mode with standard parameters")
        
        X_tr_tune, X_val_tune, _, y_tr_tune, y_val_tune, _ = tune_dm.get_data_split(seed=42)
        print(f"  Tuning Set Extracted -> Train: {len(X_tr_tune)}, Val: {len(X_val_tune)}")
        
        # Run the Tuners
        c_best_params = ClassicalSVMTuner.get_best_params(
            X_tr_tune, X_val_tune, y_tr_tune, y_val_tune,
            cache_key=f"classical_{mode}_baseline",
            verbose=False
        )

        q_best_params = QuantumSVMTuner.get_best_params(
            X_tr_tune, X_val_tune, y_tr_tune, y_val_tune, num_dims,
            cache_key=f"quantum_{mode}_baseline",
            verbose=False
        )
        
        print("\n  [LOCKED] Classical Parameters:", c_best_params)
        print("  [LOCKED] Quantum Parameters:", q_best_params)
        
        
        print("\n" + "="*80)
        print(" PHASE 2: EXPERIMENTAL SWEEP")
        print("="*80)
        
        # 1. Build the Feature Map ONCE for the locked hyperparameters
        # We use the Factory to handle transpilation for the Aer backend
        optimized_fm = FeatureMapFactory.build_zz_map(
            num_dims=num_dims, 
            reps=q_best_params['reps'], 
            entanglement=q_best_params['entanglement'],
            backend=self.qp.backend
        )

        # 2. Initialize the Kernel ONCE using the Full Stack (Sampler + Fidelity)
        shared_kernel = self.qp.get_kernel(optimized_fm)
        
        for value in experiment_values:
            c_data = {'acc': [], 'f1': [], 'time': []}
            q_data = {'acc': [], 'f1': [], 'time': []}
            
            print(f"\n{'='*80}")
            print(f"RUNNING EXPERIMENT: {value_name} = {value}")
            print(f"{'='*80}")
            
            is_kfold = mode in ['feature_complexity', 'margin', 'clusters', 'noise']
            
            if is_kfold:
                print(f"Executing {num_trials}-Fold Cross Validation...")
                current_dm = datasets_dict[value]
                
                def fold_generator():
                    for idx, splits in enumerate(current_dm.get_kfold_splits(k_folds=num_trials, seed=42)):
                        yield idx, splits
                
                data_iterator = fold_generator()
                
            else:
                print(f"Executing Monte Carlo Random Sub-Sampling ({num_trials} trials)...")
                current_dm = data_manager
                
                def sweep_generator():
                    for trial in range(num_trials):
                        X_pool, X_val, X_te, y_pool, y_val, y_te = current_dm.get_data_split(seed=trial)
                        
                        current_train_size = value if mode == 'size' else fixed_size
                        current_imbalance = value if mode == 'imbalance' else 0.5
                        
                        X_tr, y_tr = TrainingSampler.create_class_imbalance(
                            X_pool=X_pool, 
                            y_pool=y_pool, 
                            train_size=current_train_size, 
                            seed=trial, 
                            imbalance_ratio=current_imbalance
                        )
                        
                        yield trial, (X_tr, X_val, X_te, y_tr, y_val, y_te)
                
                data_iterator = sweep_generator()
            
            for idx, (X_tr, _, X_te, y_tr, _, y_te) in data_iterator:
                
                label = "Fold" if is_kfold else "Trial"
                print(f"\n{label} {idx+1}/{num_trials} (Train: {len(X_tr)}, Test: {len(X_te)})...")
                
                # Run Classical
                cache_key=f"classical_{mode}_baseline"
                c_score, c_f1, c_time = self.run_classical(X_tr, X_te, y_tr, y_te, params=c_best_params)
                c_data['acc'].append(c_score)
                c_data['f1'].append(c_f1)
                c_data['time'].append(c_time)
                
                # Run Quantum
                q_score, q_f1, q_time = self.run_quantum(X_tr, X_te, y_tr, y_te, kernel=shared_kernel, params=q_best_params)
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
