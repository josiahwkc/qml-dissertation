"""
SVM Hyperparameter Tuner
================================
Author: Josiah Chan (K23091949)

Description:
  ...
"""

import time
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityStatevectorKernel


class ClassicalSVMTuner:
    """Tunes Classical SVM (RBF kernel) hyperparameters using hold-out validation"""
    
    _cached_params = {}
    
    @classmethod
    def get_best_params(cls, X_train, X_val, y_train, y_val, cache_key=None, verbose=True):
        """
        Find best hyperparameters using hold-out validation.
        
        Args:
            X_train: Training features
            X_val: Validation features
            y_train: Training labels
            y_val: Validation labels
            cache_key: Unique identifier for this dataset/config
            verbose: Print tuning progress
            
        Returns:
            dict: Best hyperparameters {'C': ..., 'gamma': ...}
        """
        if cache_key and cache_key in cls._cached_params:
            if verbose:
                print(f"  [Classical Tuner] Using cached params for '{cache_key}'")
            return cls._cached_params[cache_key]
        
        if verbose:
            print(f"  [Classical Tuner] Starting hyperparameter search...")
            start_time = time.time()
        
        # Define hyperparameter grid
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 0.001, 0.01, 0.1, 1]
        }
        
        best_score = -1.0
        best_params = {}
        
        # Grid search over all combinations
        for C in param_grid['C']:
            for gamma in param_grid['gamma']:
                # Train on training set
                clf = SVC(kernel='rbf', C=C, gamma=gamma)
                clf.fit(X_train, y_train)
                
                # Evaluate on validation set
                y_val_pred = clf.predict(X_val)
                val_score = metrics.accuracy_score(y_val, y_val_pred)
                
                # Track best
                if val_score > best_score:
                    best_score = val_score
                    best_params = {'C': C, 'gamma': gamma}
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"  [Classical Tuner] Complete in {elapsed:.2f}s")
            print(f"  [Classical Tuner] Best params: C={best_params['C']}, gamma={best_params['gamma']}")
            print(f"  [Classical Tuner] Validation accuracy: {best_score:.4f}")
        
        # Cache results
        if cache_key:
            cls._cached_params[cache_key] = best_params
        
        return best_params


class QuantumSVMTuner:
    """Tunes Quantum SVM hyperparameters using hold-out validation"""
    
    _cached_params = {}
    
    @classmethod
    def get_best_params(cls, X_train, X_val, y_train, y_val, num_dims, cache_key=None, verbose=True):
        """
        Find best quantum and classical hyperparameters using hold-out validation.
        
        Args:
            X_train: Training features
            X_val: Validation features  
            y_train: Training labels
            y_val: Validation labels
            num_dims: Number of qubits / feature dimensions
            cache_key: Unique identifier for this dataset configuration
            verbose: Print tuning progress
            
        Returns:
            dict: Best hyperparameters {'reps': ..., 'entanglement': ..., 'C': ...}
        """
        if cache_key and cache_key in cls._cached_params:
            if verbose:
                print(f"  [Quantum Tuner] Using cached params for '{cache_key}'")
            return cls._cached_params[cache_key]
        
        if verbose:
            print(f"  [Quantum Tuner] Starting hyperparameter search (num_dims={num_dims})...")
            start_time = time.time()
        
        # Define hyperparameter grids
        reps_grid = [1, 2, 3]  # Circuit depth
        entanglement_grid = ['linear', 'full']  # Qubit connectivity
        c_grid = [0.1, 1, 10, 100]  # SVM regularization
        
        best_score = -1.0
        best_params = {}
        
        # Grid search over quantum circuit configurations
        for reps in reps_grid:
            for entanglement in entanglement_grid:
                if verbose:
                    print(f"    -> Testing: reps={reps}, entanglement='{entanglement}'")
                
                # Build quantum kernel for this configuration
                feature_map = ZZFeatureMap(
                    feature_dimension=num_dims, 
                    reps=reps, 
                    entanglement=entanglement
                )
                kernel = FidelityStatevectorKernel(feature_map=feature_map)
                
                # Precompute kernel matrices (expensive!)
                try:
                    matrix_train = kernel.evaluate(x_vec=X_train)
                    matrix_val = kernel.evaluate(x_vec=X_val, y_vec=X_train)
                except ValueError as e:
                    if verbose:
                        print(f"       [!] Kernel evaluation failed: {e}")
                    continue
                
                # Grid search over C with precomputed kernels
                for C in c_grid:
                    # Train on training set
                    qsvm = SVC(kernel='precomputed', C=C)
                    qsvm.fit(matrix_train, y_train)
                    
                    # Evaluate on validation set
                    y_val_pred = qsvm.predict(matrix_val)
                    val_score = metrics.accuracy_score(y_val, y_val_pred)
                    
                    # Track best
                    if val_score > best_score:
                        best_score = val_score
                        best_params = {
                            'reps': reps,
                            'entanglement': entanglement,
                            'C': C
                        }
                        if verbose:
                            print(f"       [!] New best: val_acc={val_score:.4f} (C={C})")
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"  [Quantum Tuner] Complete in {elapsed:.2f}s")
            print(f"  [Quantum Tuner] Best params: {best_params}")
            print(f"  [Quantum Tuner] Validation accuracy: {best_score:.4f}")
        
        # Cache results
        if cache_key:
            cls._cached_params[cache_key] = best_params
        
        return best_params