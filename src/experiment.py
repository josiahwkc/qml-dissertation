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

class ExperimentConfig:
    """Configuration for experiment modes"""
    
    MODES = {
        'size': {
            'x_label': 'Training Samples',
            'title_suffix': 'vs Training Size',
            'value_name': 'Training Size',
            'is_synthetic': False
        },
        'imbalance': {
            'x_label': 'Class Imbalance Ratio',
            'title_suffix': 'vs Class Imbalance',
            'value_name': 'Ratio',
            'is_synthetic': False
        },
        'feature_complexity': {
            'x_label': 'Informative Features',
            'title_suffix': 'vs Feature Complexity',
            'value_name': 'Informative Features',
            'is_synthetic': True
        },
        'margin': {
            'x_label': 'Class Separation',
            'title_suffix': 'vs Margin',
            'value_name': 'Class Separation',
            'is_synthetic': True
        },
        'clusters': {
            'x_label': 'Clusters per Class',
            'title_suffix': 'vs Clusters',
            'value_name': 'Clusters/Class',
            'is_synthetic': True
        },
        'noise': {
            'x_label': 'Label Noise',
            'title_suffix': 'vs Noise',
            'value_name': 'Noise Fraction',
            'is_synthetic': True
        }
    }
    
    @classmethod
    def get(cls, mode):
        """Get configuration for a mode"""
        if mode not in cls.MODES:
            raise ValueError(f"Invalid mode: {mode}")
        return cls.MODES[mode]


class ExperimentRunner():
    def __init__(self, quantum_provider, sweep_values_dict, num_dims, num_trials, fixed_size):
        self.qp = quantum_provider
        self.num_dims = num_dims
        self.num_trials = num_trials
        self.sweep_values_dict = sweep_values_dict
        self.fixed_size = fixed_size
        
        self.classical_clf = SVC(kernel='rbf')
        self.data_manager = None
        self.clear_results()
    
    def initialise_datasets(self, mode=None, filename=None, target_col=None):
        if mode in ['feature_complexity', 'margin', 'clusters', 'noise']:
            self.data_manager = SyntheticDataManager()
            self.data_manager.initialise_datasets(num_dims=self.num_dims, mode=mode, sweep_values=self.sweep_values_dict[mode])
        
        elif mode in ['size', 'imbalance']:
            self.data_manager = CSVDataManager()
            self.data_manager.load_dataset(filename=filename, target_col=target_col, num_dims=self.num_dims)
        
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def clear_results(self):
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
    
    def run_experiment(self, mode):
        """
        Run complete experiment pipeline.
        
        Clean orchestration method that delegates to focused helper methods.
        """
        
        # Resets and clears stored results
        self.clear_results()
           
        config = ExperimentConfig.get(mode)
        
        sweep_values = self.sweep_values_dict[mode]
        
        # Phase 1: Tune hyperparameters  
        c_params, q_params = self._tune_hyperparameters(mode, sweep_values)

        # Phase 2: Build quantum kernel once
        # shared_kernel = self._build_quantum_kernel(q_params)
        feature_map = ZZFeatureMap(
            feature_dimension=self.num_dims,
            reps=q_params['reps'],
            entanglement=q_params['entanglement'],
        )
        shared_kernel = FidelityQuantumKernel(feature_map=feature_map)
        
        
        # Phase 3: Run trials for each sweep value
        print("\n" + "="*80)
        print(" PHASE 3: RUNNING SWEEPS")
        print("="*80)
        for value in sweep_values:
            self._run_trials_for_value(
                mode=mode,
                value=value,
                config=config,
                num_trials=self.num_trials,
                fixed_size=self.fixed_size,
                c_params=c_params,
                q_params=q_params,
                shared_kernel=shared_kernel
            )
        
        # Phase 4: Plot results after all values complete
        self.plot_results(config['x_label'], config['title_suffix'])
    
    def _tune_hyperparameters(self, mode, sweep_values):
        """
        Tune hyperparameters on baseline dataset.
        
        Returns:
            tuple: (classical_params, quantum_params)
        """
        print("\n" + "="*80)
        print(" PHASE 1: HYPERPARAMETER OPTIMIZATION")
        print("="*80)
        
        # Get baseline dataset
        X_train, X_val, y_train, y_val = self._get_baseline_split(mode, sweep_values)
        
        print(f"  Tuning set: Train={len(y_train)}, Val={len(y_val)}")
        
        # Tune classical
        c_params = ClassicalSVMTuner.get_best_params(
            X_train, X_val, y_train, y_val,
            cache_key=f"classical_{mode}_baseline",
            verbose=False
        )
        
        # Tune quantum
        q_params = QuantumSVMTuner.get_best_params(
            X_train, X_val, y_train, y_val, self.num_dims,
            cache_key=f"quantum_{mode}_baseline",
            verbose=False
        )
        
        print("\n" + "-"*80)
        print("  [LOCKED] Classical:", c_params)
        print("  [LOCKED] Quantum:", q_params)
        print("-"*80)
        
        return c_params, q_params
    
    def _get_baseline_split(self, mode, sweep_values):
        """Get train/val split from baseline dataset"""
        config = ExperimentConfig.get(mode)
        
        if config['is_synthetic']:
            # Use middle value as baseline
            baseline_value = sweep_values[len(sweep_values) // 2]
            label = f"{mode}_{baseline_value}"
            X_train, X_val, _, y_train, y_val, _ = self.data_manager.get_data_split(seed=42, label=label)
        else:
            # Use entire dataset
            X_train, X_val, _, y_train, y_val, _ = self.data_manager.get_data_split(seed=42)
        
        return X_train, X_val, y_train, y_val
    
    def _build_quantum_kernel(self, q_params):
        """Build quantum kernel with tuned parameters"""
        print("\n" + "="*80)
        print(" PHASE 2: BUILDING QUANTUM KERNEL")
        print("="*80)
        
        optimized_fm = FeatureMapFactory.build_zz_map(
            num_dims=self.num_dims,
            reps=q_params['reps'],
            entanglement=q_params['entanglement'],
            backend=self.qp.backend
        )
        
        kernel = self.qp.get_kernel(optimized_fm)
        
        print(f"  Kernel built: reps={q_params['reps']}, "
        f"entanglement={q_params['entanglement']}")
        
        return kernel

    def _run_trials_for_value(self, mode, value, config, num_trials, 
                              fixed_size, c_params, q_params, shared_kernel):
        """Run all trials for a single sweep value"""
        print(f"\n{'-'*80}")
        print(f"RUNNING: {config['value_name']} = {value}")
        print(f"{'-'*80}")
        
        # Get data iterator
        data_iterator = self._get_data_iterator(
            mode, value, config, num_trials, fixed_size
        )
        
        # Run trials and collect results
        c_data, q_data = self._run_trials(
            data_iterator, num_trials, config['is_synthetic'],
            c_params, q_params, shared_kernel
        )
        
        # Calculate and display statistics
        self._process_results(
            value, c_data, q_data, num_trials, config
        )
    
    def _get_data_iterator(self, mode, value, config, num_trials, fixed_size):
        """Get appropriate data iterator based on mode"""
        if config['is_synthetic']:
            return self._get_kfold_iterator(mode, value, num_trials)
        else:
            return self._get_monte_carlo_iterator(
                mode, value, num_trials, fixed_size
            )
    
    def _get_kfold_iterator(self, mode, value, num_trials):
        """K-fold cross-validation iterator for synthetic data"""
        print(f"Executing {num_trials}-Fold Cross Validation...")
        
        label = f"{mode}_{value}"
        
        for idx, splits in enumerate(self.data_manager.get_kfold_splits(k_folds=num_trials, seed=42, label=label)):
            yield idx, splits
    
    def _get_monte_carlo_iterator(self, mode, value, num_trials, fixed_size):
        """Monte Carlo random sub-sampling for CSV data"""
        print(f"Executing Monte Carlo Sub-Sampling ({num_trials} trials)...")
        
        for trial in range(num_trials):
            X_pool, X_val, X_test, y_pool, y_val, y_test = \
                self.data_manager.get_data_split(seed=trial)
            
            train_size = value if mode == 'size' else fixed_size
            imbalance = value if mode == 'imbalance' else 0.5
            
            X_train, y_train = TrainingSampler.create_class_imbalance(
                X_pool=X_pool,
                y_pool=y_pool,
                train_size=train_size,
                seed=trial,
                imbalance_ratio=imbalance
            )
            
            yield trial, (X_train, X_val, X_test, y_train, y_val, y_test)
    
    def _run_trials(self, data_iterator, num_trials, is_kfold, 
                   c_params, q_params, shared_kernel):
        """Execute all trials and collect results"""
        c_data = {'acc': [], 'f1': [], 'time': []}
        q_data = {'acc': [], 'f1': [], 'time': []}
        
        label_type = "Fold" if is_kfold else "Trial"
        
        for idx, (X_train, _, X_test, y_train, _, y_test) in data_iterator:
            print(f"\n{label_type} {idx+1}/{num_trials} "
                  f"(Train: {len(X_train)}, Test: {len(X_test)})...")
            
            # Run classical
            c_acc, c_f1, c_time = self.run_classical(
                X_train, X_test, y_train, y_test, params=c_params
            )
            c_data['acc'].append(c_acc)
            c_data['f1'].append(c_f1)
            c_data['time'].append(c_time)
            
            # Run quantum
            q_acc, q_f1, q_time = self.run_quantum(
                X_train, X_test, y_train, y_test,
                kernel=shared_kernel, params=q_params
            )
            q_data['acc'].append(q_acc)
            q_data['f1'].append(q_f1)
            q_data['time'].append(q_time)
        
        return c_data, q_data

    def _process_results(self, value, c_data, q_data, num_trials, config):
        """Calculate statistics, print tables, and store results"""
        # Calculate statistics
        stats = self._calculate_statistics(c_data, q_data)
        
        # Print performance tables
        self._print_results_tables(c_data, q_data, stats, num_trials)
        
        # Store for plotting
        self._store_results(value, stats)
    
    def _calculate_statistics(self, c_data, q_data):
        """Calculate all statistics for comparison"""
        # Get last trial's train/test sizes
        n_train = len(c_data['acc'])  # Placeholder - get from actual data
        n_test = len(c_data['acc'])   # Placeholder - get from actual data
        
        stats = {
            # Classical
            'c_avg_acc': np.mean(c_data['acc']),
            'c_std_acc': np.std(c_data['acc']),
            'c_avg_f1': np.mean(c_data['f1']),
            'c_std_f1': np.std(c_data['f1']),
            'c_avg_time': np.mean(c_data['time']),
            
            # Quantum
            'q_avg_acc': np.mean(q_data['acc']),
            'q_std_acc': np.std(q_data['acc']),
            'q_avg_f1': np.mean(q_data['f1']),
            'q_std_f1': np.std(q_data['f1']),
            'q_avg_time': np.mean(q_data['time']),
        }
        
        # Deltas
        stats['delta_acc'] = stats['q_avg_acc'] - stats['c_avg_acc']
        stats['delta_f1'] = stats['q_avg_f1'] - stats['c_avg_f1']
        stats['time_ratio'] = (stats['q_avg_time'] / stats['c_avg_time'] 
                              if stats['c_avg_time'] > 0 else stats['q_avg_time'])
        
        # Effect sizes (Cohen's d)
        std_pooled_acc = np.sqrt((stats['q_std_acc']**2 + stats['c_std_acc']**2) / 2)
        stats['cohen_d_acc'] = (stats['delta_acc'] / std_pooled_acc 
                               if std_pooled_acc > 0 else 0.0)
        
        std_pooled_f1 = np.sqrt((stats['q_std_f1']**2 + stats['c_std_f1']**2) / 2)
        stats['cohen_d_f1'] = (stats['delta_f1'] / std_pooled_f1 
                              if std_pooled_f1 > 0 else 0.0)
        
        # Statistical tests
        stats['p_val_acc'] = nadeau_bengio_corrected_ttest(
            q_data['acc'], c_data['acc'], n_train, n_test
        )
        stats['p_val_f1'] = nadeau_bengio_corrected_ttest(
            q_data['f1'], c_data['f1'], n_train, n_test
        )
        
        return stats
    
    def _print_results_tables(self, c_data, q_data, stats, num_trials):
        """Print formatted results tables"""
        # Table 1: Quantum performance
        self._print_model_table("QUANTUM", q_data, stats, num_trials, is_quantum=True)
        
        # Table 2: Classical performance
        self._print_model_table("CLASSICAL", c_data, stats, num_trials, is_quantum=False)
        
        # Table 3: Statistical comparison
        self._print_comparison_table(stats)
    
    def _print_model_table(self, model_name, data, stats, num_trials, is_quantum):
        """Print performance table for one model"""
        prefix = 'q' if is_quantum else 'c'
        
        print(f"\n[ {model_name} MODEL PERFORMANCE ]")
        print("-" * 50)
        print(f"{'Trial':<6} | {'Acc':<10} | {'F1':<10} | {'Time (s)':<10}")
        print("-" * 50)
        
        for i in range(num_trials):
            print(f"{i+1:<6} | {data['acc'][i]:<10.2%} | "
                  f"{data['f1'][i]:<10.2f} | {data['time'][i]:<10.4f}")
        
        print("-" * 50)
        print(f"{'AVG':<6} | {stats[f'{prefix}_avg_acc']:<10.2%} | "
              f"{stats[f'{prefix}_avg_f1']:<10.2f} | {stats[f'{prefix}_avg_time']:<10.4f}")
        print(f"{'STD':<6} | {stats[f'{prefix}_std_acc']:<10.2%} | "
              f"{stats[f'{prefix}_std_f1']:<10.2f} | {'-':<10}")
        print("-" * 50)
    
    def _print_comparison_table(self, stats):
        """Print statistical comparison table"""
        print("\n[ STATISTICAL ANALYSIS ]")
        print("-" * 85)
        print(f'{"Metric":<8} | {"Delta":<12} | {"Cohen\'s d":<10} | '
              f'{"p-value":<10} | {"Significant?"}')
        print("-" * 85)
        
        sig_acc = "YES (*)" if stats['p_val_acc'] < 0.05 else "NO"
        sig_f1 = "YES (*)" if stats['p_val_f1'] < 0.05 else "NO"
        
        print(f"{'Accuracy':<8} | {stats['delta_acc']:>+12.2%} | "
              f"{stats['cohen_d_acc']:>10.2f} | {stats['p_val_acc']:>10.4f} | {sig_acc}")
        print(f"{'F1 Score':<8} | {stats['delta_f1']:>+12.2%} | "
              f"{stats['cohen_d_f1']:>10.2f} | {stats['p_val_f1']:>10.4f} | {sig_f1}")
        print(f"{'Time':<8} | QSVM was {stats['time_ratio']:.0f}x slower")
        print("-" * 85)
        print()
    
    def _store_results(self, value, stats):
        """Store results for plotting"""
        self.results['x_values'].append(value)
        
        # Classical
        self.results['c_acc'].append(stats['c_avg_acc'])
        self.results['c_acc_std'].append(stats['c_std_acc'])
        self.results['c_f1'].append(stats['c_avg_f1'])
        self.results['c_f1_std'].append(stats['c_std_f1'])
        self.results['c_time'].append(stats['c_avg_time'])
        
        # Quantum
        self.results['q_acc'].append(stats['q_avg_acc'])
        self.results['q_acc_std'].append(stats['q_std_acc'])
        self.results['q_f1'].append(stats['q_avg_f1'])
        self.results['q_f1_std'].append(stats['q_std_f1'])
        self.results['q_time'].append(stats['q_avg_time'])
        
        # Comparisons
        self.results['delta_acc'].append(stats['delta_acc'])
        self.results['p_val_acc'].append(stats['p_val_acc'])
        self.results['delta_f1'].append(stats['delta_f1'])
        self.results['p_val_f1'].append(stats['p_val_f1'])

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
