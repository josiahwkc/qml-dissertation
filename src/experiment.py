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
    def __init__(self, num_dims=4):
        self.num_dims = num_dims
        digits = datasets.load_digits(n_class=2)
        self.X = digits.data
        self.y = digits.target
        
    def get_data_split(self, train_size, seed):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, train_size=train_size, random_state=seed, stratify=self.y
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
            'sizes': [],
            'q_acc': [], 'q_f1': [], 'q_time': [],
            'c_acc': [], 'c_f1': [], 'c_time': []
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

# %%
# Main execution

TRAINING_SIZES = [20, 40]
# TRAINING_SIZES = [20, 40, 60, 80, 100]
NUM_TRIALS = 3
NUM_DIMS = 4

data_manager = DataManager(num_dims=NUM_DIMS)
runner = ExperimentRunner()

# Table Header
print("-" * 80)
header = f"{'Size':<6} | {'Trial':<5} | {'C-Acc':<8} | {'C-F1':<8} | {'C-Time':<9} || {'Q-Acc':<8} | {'Q-F1':<8} | {'Q-Time':<9}"
print(header)
print("-" * 80)

for size in TRAINING_SIZES:
    # Temporary lists for this size iteration
    c_accs, c_f1s, c_times = [], [], []
    q_accs, q_f1s, q_times = [], [], []
    
    for seed in range(NUM_TRIALS):
        # Get Data
        X_tr, X_te, y_tr, y_te = data_manager.get_data_split(size, seed)
        
        # Run Classical
        c_score, c_f1, c_time = runner.run_classical(X_tr, X_te, y_tr, y_te)
        c_accs.append(c_score)
        c_f1s.append(c_f1)
        c_times.append(c_time)
        
        # Run Quantum
        q_score, q_f1, q_time = runner.run_quantum(X_tr, X_te, y_tr, y_te, NUM_DIMS)
        q_accs.append(q_score)
        q_f1s.append(q_f1)
        q_times.append(q_time)
        
        # Print Row
        row = (f"{size:<6} | {seed+1:<5} | "
               f"{c_score:<8.2%} | {c_f1:<8.2f} | {c_time:<9.4f} || "
               f"{q_score:<8.2%} | {q_f1:<8.2f} | {q_time:<9.4f}")
        print(row)

    avg_c_acc = np.mean(c_accs)
    avg_c_f1 = np.mean(c_f1s)
    avg_c_time = np.mean(c_times)
    
    avg_q_acc = np.mean(q_accs)
    avg_q_f1 = np.mean(q_f1s)
    avg_q_time = np.mean(q_times)

    runner.results['sizes'].append(size)
    runner.results['c_acc'].append(avg_c_acc)
    runner.results['c_f1'].append(avg_c_f1)
    runner.results['c_time'].append(avg_c_time)
    
    runner.results['q_acc'].append(avg_q_acc)
    runner.results['q_f1'].append(avg_q_f1)
    runner.results['q_time'].append(avg_q_time)
    
    # Print Summary Block
    print("-" * 105)
    print(f"SUMMARY FOR SIZE {size} (Average of {NUM_TRIALS} trials):")
    print(f"Classical > Acc: {avg_c_acc:.2%} | F1: {avg_c_f1:.2f} | Time: {avg_c_time:.4f}s")
    print(f"Quantum   > Acc: {avg_q_acc:.2%} | F1: {avg_q_f1:.2f} | Time: {avg_q_time:.4f}s")
    print("-" * 105)

# %%
# VISUALIZATION
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Accuracy
ax1.plot(runner.results['sizes'], runner.results['c_acc'], 'o-', label='Classical', color='blue')
ax1.plot(runner.results['sizes'], runner.results['q_acc'], 's-', label='Quantum', color='purple')
ax1.set_title('Accuracy')
ax1.set_xlabel('Training Samples')
ax1.legend()
ax1.grid(True)

# F1 Score
ax2.plot(runner.results['sizes'], runner.results['c_f1'], 'o-', label='Classical', color='blue')
ax2.plot(runner.results['sizes'], runner.results['q_f1'], 's-', label='Quantum', color='purple')
ax2.set_title('F1 Score (Weighted)')
ax2.set_xlabel('Training Samples')
ax2.legend()
ax2.grid(True)

# Time
ax3.plot(runner.results['sizes'], runner.results['c_time'], 'o-', label='Classical', color='blue')
ax3.plot(runner.results['sizes'], runner.results['q_time'], 's-', label='Quantum', color='purple')
ax3.set_title('Training Time (Log Scale)')
ax3.set_yscale('log')
ax3.set_xlabel('Training Samples')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.show()