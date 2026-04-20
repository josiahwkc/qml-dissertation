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

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVC
from sklearn import metrics

from data_manager import CSVDataManager, QuantumBenchmarkDataManager, SyntheticDataManager, TrainingSampler
from tuner import ClassicalSVMTuner, QuantumSVMTuner

# Quantum Imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from feature_map_factory import FeatureMapFactory

class ExperimentConfig:
    """Configuration for experiment modes with explicit data source types"""
    
    NUM_DIMS = 5          # Number of PCA dimensions (and Qubits)
    NUM_TRIALS = 10 #30   # Number of random seeds to average over
    N_CLASS = 2           # Binary classification
    FIXED_SIZE = 100      # Default training size for non-'size' experiments
    
    # Define data source types
    DATA_SOURCE_CSV = 'csv'
    DATA_SOURCE_SKLEARN = 'sklearn'
    DATA_SOURCE_QISKIT = 'qiskit'
    
    MODES = {
        # =====================================================================
        # CSV DATA MODES (Real datasets)
        # =====================================================================
        'size': {
            'x_label': 'Training Samples',
            'title_suffix': 'vs Training Size',
            'value_name': 'Training Size',
            'data_source': DATA_SOURCE_CSV,
            'requires_file': True,
            'sweep_values': [50, 100, 150, 200, 250, 300],
            'description': 'Vary training set size on real data'
        },
        'imbalance': {
            'x_label': 'Class Imbalance Ratio',
            'title_suffix': 'vs Class Imbalance',
            'value_name': 'Ratio',
            'data_source': DATA_SOURCE_CSV,
            'requires_file': True,
            'sweep_values': [0.5, 0.6, 0.7, 0.8, 0.9],
            'description': 'Vary class imbalance ratio on real data'
        },
        
        # =====================================================================
        # SKLEARN SYNTHETIC MODES (make_classification, make_blobs)
        # =====================================================================
        'feature_complexity': {
            'x_label': 'Informative Features',
            'title_suffix': 'vs Feature Complexity',
            'value_name': 'Informative Features',
            'data_source': DATA_SOURCE_SKLEARN,
            'requires_file': False,
            'sweep_values': [1, 2, 3, 4, 5],
            'sweep_parameter': 'n_informative',
            'description': 'Vary number of informative features in synthetic data'
        },
        'margin': {
            'x_label': 'Class Separation',
            'title_suffix': 'vs Margin',
            'value_name': 'Class Separation',
            'data_source': DATA_SOURCE_SKLEARN,
            'requires_file': False,
            'sweep_values': [0.1, 0.5, 1.0, 1.5, 2.0],
            'sweep_parameter': 'class_sep',
            'description': 'Vary class separation margin in synthetic data'
        },
        'clusters': {
            'x_label': 'Clusters per Class',
            'title_suffix': 'vs Clusters',
            'value_name': 'Clusters/Class',
            'data_source': DATA_SOURCE_SKLEARN,
            'requires_file': False,
            'sweep_values': [1, 2, 3, 4],
            'sweep_parameter': 'n_clusters_per_class',
            'description': 'Vary decision boundary complexity via clusters'
        },
        'noise': {
            'x_label': 'Label Noise',
            'title_suffix': 'vs Noise',
            'value_name': 'Noise Fraction',
            'data_source': DATA_SOURCE_SKLEARN,
            'requires_file': False,
            'sweep_values': [0.0, 0.05, 0.10, 0.15, 0.20],
            'sweep_parameter': 'flip_y',
            'description': 'Vary label noise in synthetic data'
        },
        'inter_distance': {
            'x_label': 'Inter-class Distance',
            'title_suffix': 'vs Inter-class Distance',
            'value_name': 'Centroid Distance',
            'data_source': DATA_SOURCE_SKLEARN,
            'requires_file': False,
            'sweep_values': [1.4, 1.2, 1.0, 0.8, 0.6],  # Decreasing distance = higher entanglement
            'sweep_parameter': 'inter_distance',
            'description': 'Vary the spatial distance between class centroids'
        },
        'intra_spread': {
            'x_label': 'Intra-class Variance (Std Dev)',
            'title_suffix': 'vs Intra-class Variance',
            'value_name': 'Cluster Spread',
            'data_source': DATA_SOURCE_SKLEARN, 
            'requires_file': False,
            'sweep_values': [0.5, 1.0, 1.5, 2.0, 3.0],  # Increasing spread = higher entanglement
            'sweep_parameter': 'cluster_std',
            'description': 'Vary the spatial scatter of points within each class'
        },
        
        # =====================================================================
        # QISKIT SYNTHETIC MODE (ad_hoc_data)
        # =====================================================================
        'quantum_benchmark': {
            'x_label': 'Training Samples',
            'title_suffix': 'vs Training Size',
            'value_name': 'Training Size',
            'data_source': DATA_SOURCE_QISKIT,
            'requires_file': False,
            'sweep_values': [50, 100, 150, 200, 250, 300],
            'sweep_parameter': 'gap',
            'description': 'Qiskit ad_hoc_data optimized for ZZFeatureMap',
            'fixed_dims': 2,  # ad_hoc_data only allows 2 or 3 dimensions
        },
    }
    
    @classmethod
    def get(cls, mode):
        """
        Get configuration for a mode.
        
        Args:
            mode: Experiment mode name
            
        Returns:
            dict: Configuration dictionary
            
        Raises:
            ValueError: If mode is invalid
        """
        if mode not in cls.MODES:
            available = ', '.join(cls.MODES.keys())
            raise ValueError(
                f"Invalid mode: '{mode}'. Available modes: {available}"
            )
        return cls.MODES[mode]


class ExperimentRunner():
    def __init__(self, quantum_provider):
        self.qp = quantum_provider
        self.num_dims = ExperimentConfig.NUM_DIMS
        self.num_trials = ExperimentConfig.NUM_TRIALS
        self.fixed_size = ExperimentConfig.FIXED_SIZE
        self.config = None
        self.data_manager = None
        self.clear_results()
        
    def initialise_datasets(self, mode, filename=None, target_col=None):
        """
        Initialize datasets based on mode's data source.
        
        Args:
            mode: Experiment mode
            filename: CSV filename (required for CSV modes)
            target_col: Target column (required for CSV modes)
        """
        
        # Validation
        if self.config['requires_file'] and (filename is None or target_col is None):
            raise ValueError(
                f"Mode '{mode}' requires filename and target_col parameters"
            )
        
        data_source = self.config['data_source']
        sweep_values = self.config['sweep_values']
        
        if data_source == ExperimentConfig.DATA_SOURCE_CSV:
            print(f"Loading CSV dataset: {filename}")
            self.data_manager = CSVDataManager()
            self.data_manager.load_dataset(
                filename=filename,
                target_col=target_col,
                num_dims=self.num_dims
            )
        
        elif data_source == ExperimentConfig.DATA_SOURCE_SKLEARN:
            print(f"Generating sklearn synthetic datasets for '{mode}' mode...")
            self.data_manager = SyntheticDataManager()
            self.data_manager.initialise_datasets(
                num_dims=self.num_dims,
                mode=mode,
                sweep_values=sweep_values
            )
        
        elif data_source == ExperimentConfig.DATA_SOURCE_QISKIT:
            print(f"Generating Qiskit ad_hoc datasets for '{mode}' mode...")
            
            fixed_dims = self.config['fixed_dims']
            print(f"Note: ad_hoc_data works best with {fixed_dims} dimensions.")
            print(f"Generating dataset with {fixed_dims} dimensions.")
                
            self.data_manager = QuantumBenchmarkDataManager()
            self.data_manager.create_dataset()
        
        else:
            raise ValueError(f"Unknown data source: {data_source}")
    
    def clear_results(self):
        self.results = {
            'x_values': [],
            'q_acc': [], 'q_acc_std': [], 'q_f1': [], 'q_f1_std': [], 'q_time': [],
            'c_acc': [], 'c_acc_std': [], 'c_f1': [], 'c_f1_std': [], 'c_time': [],
            'delta_acc': [], 'p_val_acc': [], 'cohen_d_acc': [],
            'delta_f1': [], 'p_val_f1': [], 'cohen_d_f1': [],
            'time_ratio': []
        }
       
    def run_classical(self, X_train, X_test, y_train, y_test, kernel, params):
        """Runs Classical SVM (RBF Kernel) with locked parameters"""            
        start_time = time.time()
        kernel.fit(X_train, y_train)
        end_time = time.time()
        
        y_pred = kernel.predict(X_test)
        
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
    
    def run_experiment(self, mode, filename=None, target_col=None):
        """
        Run complete experiment pipeline.
        
        Clean orchestration method that delegates to focused helper methods.
        """
        
        # Resets and clears stored results
        self.clear_results()
        
        self.config = ExperimentConfig.get(mode)
        self.initialise_datasets(mode, filename, target_col)
        
        
        # Phase 1: Tune hyperparameters  
        c_params, q_params = self._tune_hyperparameters(mode)

        # Phase 2: Build kernels once
        c_kernel = SVC(
            kernel='rbf',
            C=c_params['C'],
            gamma=c_params['gamma']
        )
        
        # q_kernel = self._build_quantum_kernel(q_params)
        feature_map = ZZFeatureMap(
            feature_dimension=self.num_dims,
            reps=q_params['reps'],
            entanglement=q_params['entanglement'],
        )
        q_kernel = FidelityQuantumKernel(feature_map=feature_map)
        
        
        # Phase 3: Run trials for each sweep value
        sweep_values = self.config['sweep_values']
        print("\n" + "="*80)
        print(" PHASE 3: RUNNING SWEEPS")
        print("="*80)
        for value in sweep_values:
            self._run_trials_for_value(
                mode=mode,
                value=value,
                c_params=c_params,
                q_params=q_params,
                c_kernel=c_kernel,
                q_kernel=q_kernel
            )
        
        # Phase 4: Plot results after all values complete
        self.plot_results()
        save_dir, _ = self._get_output_meta()
        
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(save_dir, 'summary_metrics.csv'), index=False)
        print(f"Data saved to {save_dir}/summary_metrics.csv")
    
    def plot_results(self):
        """Generate, save, and display individual comparison plots"""
        
        save_dir, dataset_name = self._get_output_meta()
        
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nSaving plots to: {save_dir}/")
        
        
        x_label = self.config['x_label']
        title_suffix = self.config['title_suffix']
        
        # 1. Accuracy Plot
        plt.figure(figsize=(8, 6))
        plt.errorbar(self.results['x_values'], self.results['c_acc'], 
                     yerr=self.results['c_acc_std'], fmt='o-', capsize=5, 
                     label='Classical', color='blue')
        plt.errorbar(self.results['x_values'], self.results['q_acc'], 
                     yerr=self.results['q_acc_std'], fmt='s-', capsize=5, 
                     label='Quantum', color='purple')
        
        plt.title(f'Accuracy {title_suffix} ({dataset_name})')
        plt.xlabel(x_label)
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, 'accuracy.pdf'), format='pdf', dpi=300)
        plt.show()
        
        # 2. F1 Score Plot
        plt.figure(figsize=(8, 6))
        plt.errorbar(self.results['x_values'], self.results['c_f1'], 
                     yerr=self.results['c_f1_std'], fmt='o-', capsize=5, 
                     label='Classical', color='blue')
        plt.errorbar(self.results['x_values'], self.results['q_f1'], 
                     yerr=self.results['q_f1_std'], fmt='s-', capsize=5, 
                     label='Quantum', color='purple')
        
        plt.title(f'F1 Score {title_suffix} ({dataset_name})')
        plt.xlabel(x_label)
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, 'f1_score.pdf'), format='pdf', dpi=300)
        plt.show()
        
        # 3. Training Time Plot
        plt.figure(figsize=(8, 6))
        plt.plot(self.results['x_values'], self.results['c_time'], 
                 'o-', label='Classical', color='blue')
        plt.plot(self.results['x_values'], self.results['q_time'], 
                 's-', label='Quantum', color='purple')
        
        plt.title(f'Training Time {title_suffix} ({dataset_name})')
        plt.yscale('log')
        plt.xlabel(x_label)
        plt.ylabel('Time (s)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(save_dir, 'training_time.pdf'), format='pdf', dpi=300)
        plt.show()
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Summary Results: {dataset_name} {title_suffix}', fontsize=16)

        # Subplot 1: Accuracy
        axes[0].errorbar(self.results['x_values'], self.results['c_acc'], yerr=self.results['c_acc_std'], 
                         fmt='o-', capsize=5, label='Classical', color='blue')
        axes[0].errorbar(self.results['x_values'], self.results['q_acc'], yerr=self.results['q_acc_std'], 
                         fmt='s-', capsize=5, label='Quantum', color='purple')
        axes[0].set_title('Accuracy')
        axes[0].set_ylabel('Accuracy')

        # Subplot 2: F1 Score
        axes[1].errorbar(self.results['x_values'], self.results['c_f1'], yerr=self.results['c_f1_std'], 
                         fmt='o-', capsize=5, label='Classical', color='blue')
        axes[1].errorbar(self.results['x_values'], self.results['q_f1'], yerr=self.results['q_f1_std'], 
                         fmt='s-', capsize=5, label='Quantum', color='purple')
        axes[1].set_title('F1 Score')
        axes[1].set_ylabel('F1 Score')

        # Subplot 3: Training Time
        axes[2].plot(self.results['x_values'], self.results['c_time'], 'o-', label='Classical', color='blue')
        axes[2].plot(self.results['x_values'], self.results['q_time'], 's-', label='Quantum', color='purple')
        axes[2].set_title('Training Time')
        axes[2].set_ylabel('Time (s)')
        axes[2].set_yscale('log')

        # Formatting for all subplots
        for ax in axes:
            ax.set_xlabel(x_label)
            ax.legend()
            ax.grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for the suptitle
        plt.savefig(os.path.join(save_dir, 'combined_results.pdf'), format='pdf', dpi=300)
        plt.show()
        
    def _tune_hyperparameters(self, mode):
        """
        Tune hyperparameters on baseline dataset.
        
        Returns:
            tuple: (classical_params, quantum_params)
        """
        print("\n" + "="*80)
        print(" PHASE 1: HYPERPARAMETER OPTIMIZATION")
        print("="*80)
        
        # Get baseline dataset
        X_train, X_val, y_train, y_val = self._get_baseline_split(mode)
        
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
    
    def _get_baseline_split(self, mode):
        """Get train/val split from baseline dataset"""
        sweep_values = self.config['sweep_values']
        data_source = self.config['data_source']
        
        if data_source == ExperimentConfig.DATA_SOURCE_SKLEARN:
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
            sampler=self.qp.sampler
        )
        
        kernel = self.qp.get_kernel(optimized_fm)
        
        print(f"  Kernel built: reps={q_params['reps']}, "
        f"entanglement={q_params['entanglement']}")
        
        return kernel

    def _run_trials_for_value(self, mode, value, c_params, q_params, c_kernel, q_kernel):
        """Run all trials for a single sweep value"""
        print(f"\n{'-'*80}")
        print(f"RUNNING: {self.config['value_name']} = {value}")
        print(f"{'-'*80}")
        
        # Get data iterator
        data_iterator = self._get_data_iterator(mode, value)
        
        # Run trials and collect results
        c_data, q_data = self._run_trials(data_iterator, c_params, q_params, c_kernel, q_kernel)
        
        # Calculate and display statistics
        self._process_results(
            value, c_data, q_data, mode
        )
    
    def _get_data_iterator(self, mode, value):
        """Get appropriate data iterator based on mode"""
        data_source = self.config['data_source']
        
        if data_source == ExperimentConfig.DATA_SOURCE_SKLEARN:
            return self._get_kfold_iterator(mode, value)
        else:
            return self._get_monte_carlo_iterator(mode, value)
    
    def _get_kfold_iterator(self, mode, value):
        """K-fold cross-validation iterator for synthetic data"""
        print(f"Executing {self.num_trials}-Fold Cross Validation...")
        
        label = f"{mode}_{value}"
        
        for idx, splits in enumerate(self.data_manager.get_kfold_splits(k_folds=self.num_trials, seed=42, label=label)):
            yield idx, splits
    
    def _get_monte_carlo_iterator(self, mode, value):
        """Monte Carlo random sub-sampling for CSV data"""
        print(f"Executing Monte Carlo Sub-Sampling ({self.num_trials} trials)...")
        
        for trial in range(self.num_trials):
            X_pool, X_val, X_test, y_pool, y_val, y_test = self.data_manager.get_data_split(seed=trial)
            
            train_size = value if mode in ['size', 'quantum_benchmark'] else self.fixed_size
            imbalance = value if mode == 'imbalance' else 0.5
            
            X_train, y_train = TrainingSampler.create_class_imbalance(
                X_pool=X_pool,
                y_pool=y_pool,
                train_size=train_size,
                seed=trial,
                imbalance_ratio=imbalance
            )
            
            yield trial, (X_train, X_val, X_test, y_train, y_val, y_test)
    
    def _run_trials(self, data_iterator, c_params, q_params, c_kernel, q_kernel):
        """Execute all trials and collect results"""
        c_data = {'acc': [], 'f1': [], 'time': []}
        q_data = {'acc': [], 'f1': [], 'time': []}
        
        is_kfold = self.config['data_source'] == ExperimentConfig.DATA_SOURCE_SKLEARN
        
        label_type = "Fold" if is_kfold else "Trial"
        
        for idx, (X_train, _, X_test, y_train, _, y_test) in data_iterator:
            print(f"\n{label_type} {idx+1}/{self.num_trials} "
                  f"(Train: {len(X_train)}, Test: {len(X_test)})...")
            
            # Run classical
            c_acc, c_f1, c_time = self.run_classical(
                X_train, X_test, y_train, y_test, 
                kernel=c_kernel, params=c_params
            )
            c_data['acc'].append(c_acc)
            c_data['f1'].append(c_f1)
            c_data['time'].append(c_time)
            
            # Run quantum
            q_acc, q_f1, q_time = self.run_quantum(
                X_train, X_test, y_train, y_test,
                kernel=q_kernel, params=q_params
            )
            q_data['acc'].append(q_acc)
            q_data['f1'].append(q_f1)
            q_data['time'].append(q_time)
        
        return c_data, q_data

    def _process_results(self, value, c_data, q_data, mode):
        """Calculate statistics, print tables, and store results"""
        # Calculate statistics
        stats = self._calculate_statistics(c_data, q_data)
        
        # Print performance tables
        self._print_results_tables(value, c_data, q_data, stats)
        
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
        stats['p_val_acc'] = self._nadeau_bengio_corrected_ttest(
            q_data['acc'], c_data['acc'], n_train, n_test
        )
        stats['p_val_f1'] = self._nadeau_bengio_corrected_ttest(
            q_data['f1'], c_data['f1'], n_train, n_test
        )
        
        return stats
    
    def _nadeau_bengio_corrected_ttest(self, q_scores, c_scores, n_train, n_test):
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
    
    def _print_results_tables(self, value, c_data, q_data, stats):
        """Generate formatted tables, print to console, and append to log file"""
        
        # Build the text report as a list of strings
        report = []
        report.append(f"\n{'='*85}")
        report.append(f"RESULTS FOR {self.config['value_name'].upper()}: {value}")
        report.append(f"{'='*85}")
        
        report.append(self._build_model_table_string("QUANTUM", q_data, stats))
        report.append(self._build_model_table_string("CLASSICAL", c_data, stats))
        report.append(self._build_comparison_table_string(stats))
        
        # Combine list into a single massive string
        full_report = "\n".join(report)
        
        # Print to console for real-time monitoring
        print(full_report)
        
        # Unpack the tuple to safely get JUST the directory path
        save_dir, _ = self._get_output_meta()
        log_file = os.path.join(save_dir, 'detailed_report.txt')
        
        # Append to detailed_report.txt in your results folder
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(full_report + "\n")

    def _build_model_table_string(self, model_name, data, stats):
        """Builds and returns the performance table string for one model"""
        prefix = 'q' if model_name == "QUANTUM" else 'c'
        lines = [
            f"\n[ {model_name} MODEL PERFORMANCE ]",
            "-" * 50,
            f"{'Trial':<6} | {'Acc':<10} | {'F1':<10} | {'Time (s)':<10}",
            "-" * 50
        ]
        
        for i in range(self.num_trials):
            lines.append(f"{i+1:<6} | {data['acc'][i]:<10.2%} | "
                         f"{data['f1'][i]:<10.2f} | {data['time'][i]:<10.4f}")
            
        lines.extend([
            "-" * 50,
            f"{'AVG':<6} | {stats[f'{prefix}_avg_acc']:<10.2%} | "
            f"{stats[f'{prefix}_avg_f1']:<10.2f} | {stats[f'{prefix}_avg_time']:<10.4f}",
            f"{'STD':<6} | {stats[f'{prefix}_std_acc']:<10.2%} | "
            f"{stats[f'{prefix}_std_f1']:<10.2f} | {'-':<10}",
            "-" * 50
        ])
        return "\n".join(lines)

    def _build_comparison_table_string(self, stats):
        """Builds and returns the statistical comparison table string"""
        sig_acc = "YES (*)" if stats['p_val_acc'] < 0.05 else "NO"
        sig_f1 = "YES (*)" if stats['p_val_f1'] < 0.05 else "NO"
        
        lines = [
            "\n[ STATISTICAL ANALYSIS ]",
            "-" * 85,
            f"{'Metric':<8} | {'Delta':<12} | {'Cohen\'s d':<10} | {'p-value':<10} | {'Significant?'}",
            "-" * 85,
            f"{'Accuracy':<8} | {stats['delta_acc']:>+12.2%} | {stats['cohen_d_acc']:>10.2f} | {stats['p_val_acc']:>10.4f} | {sig_acc}",
            f"{'F1 Score':<8} | {stats['delta_f1']:>+12.2%} | {stats['cohen_d_f1']:>10.2f} | {stats['p_val_f1']:>10.4f} | {sig_f1}",
            f"{'Time':<8} | QSVM was {stats['time_ratio']:.0f}x slower",
            "-" * 85,
            "\n"
        ]
        return "\n".join(lines)
        
    def _get_output_meta(self):
        """Helper to dynamically generate the save directory and dataset display name"""
        data_source = self.config['data_source']
        
        if data_source == ExperimentConfig.DATA_SOURCE_CSV:
            dataset_name = self.data_manager.filename
            base_folder = dataset_name.replace('.csv', '')
        elif data_source == ExperimentConfig.DATA_SOURCE_SKLEARN:
            dataset_name = "Sklearn Synthetic"
            base_folder = "sklearn_synthetic"
        elif data_source == ExperimentConfig.DATA_SOURCE_QISKIT:
            dataset_name = "Qiskit Ad Hoc"
            base_folder = "qiskit_adhoc"
        else:
            dataset_name = "Unknown Dataset"
            base_folder = "unknown"
            
        folder_name = f"{base_folder}_{self.config['value_name']}"
        safe_folder = folder_name.replace(' ', '_').lower()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        save_dir = os.path.join(parent_dir, "results", safe_folder)
        
        os.makedirs(save_dir, exist_ok=True)
        
        return save_dir, dataset_name
    
    def _store_results(self, value, stats):
        """Store results for plotting and CSV export"""
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
        self.results['cohen_d_acc'].append(stats['cohen_d_acc'])
        
        self.results['delta_f1'].append(stats['delta_f1'])
        self.results['p_val_f1'].append(stats['p_val_f1'])
        self.results['cohen_d_f1'].append(stats['cohen_d_f1'])
        
        self.results['time_ratio'].append(stats['time_ratio'])
