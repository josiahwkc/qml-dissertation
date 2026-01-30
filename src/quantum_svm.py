# %%
# IMPORTS
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# QUANTUM IMPORTS
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# %%
# 1. CONFIGURATION
TRAINING_SIZES = [20, 50, 100] 
N_TRIALS = 3  
N_COMPONENTS = 4

# %%
# 2. DATA PREPARATION
digits = datasets.load_digits()
X_all = digits.data
y_all = digits.target

binary_mask = (y_all == 0) | (y_all == 1) | (y_all == 2)
X_binary = X_all[binary_mask]
y_binary = y_all[binary_mask]

# Split raw data (Locked Test Set)
X_train_pool, X_test_fixed_raw, y_train_pool, y_test_fixed = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"Fixed Test Set Size: {len(X_test_fixed_raw)} samples")
print("-" * 50)

# %%
# 3. THE EXPERIMENT LOOP
mean_accuracies = []
std_accuracies = []
mean_f1s = []
std_f1s = []
mean_times = []

for size in TRAINING_SIZES:
    trial_accuracies = []
    trial_f1s = []
    trial_times = []
    
    print(f"Training QSVC on subset size: {size}")

    for seed in range(N_TRIALS):
        X_subset_raw, _, y_subset, _ = train_test_split(
            X_train_pool, y_train_pool, train_size=size, 
            random_state=seed, stratify=y_train_pool
        )
        
        preprocessor = Pipeline([
            ('scaler_std', StandardScaler()),
            ('pca', PCA(n_components=N_COMPONENTS)),
            ('scaler_minmax', MinMaxScaler(feature_range=(-1, 1)))
        ])
        
        preprocessor.fit(X_subset_raw)
        X_subset_transformed = preprocessor.transform(X_subset_raw)
        X_test_transformed = preprocessor.transform(X_test_fixed_raw)
        
        feature_map = ZZFeatureMap(
            feature_dimension=N_COMPONENTS, 
            reps=2, 
            entanglement='linear'
        )
        
        kernel = FidelityQuantumKernel(feature_map=feature_map)
        qsvc = QSVC(quantum_kernel=kernel)
        
        start_time = time.time()
        qsvc.fit(X_subset_transformed, y_subset)
        end_time = time.time()
        
        trial_times.append(end_time - start_time)
        
        y_pred = qsvc.predict(X_test_transformed)
        
        score = metrics.accuracy_score(y_test_fixed, y_pred)
        trial_accuracies.append(score)
        
        f1 = f1_score(y_test_fixed, y_pred, average='weighted')
        trial_f1s.append(f1)
        
        print(f"Trial {seed+1}/{N_TRIALS}: Acc={score:.2%} | Time={trial_times[-1]:.2f}s | F1={trial_f1s[-1]:.2f}")

    # Average results
    avg_acc = np.mean(trial_accuracies)
    std_dev = np.std(trial_accuracies)
    
    avg_f1 = np.mean(trial_f1s)
    std_f1 = np.std(trial_f1s)
    
    avg_time = np.mean(trial_times)
    
    mean_accuracies.append(avg_acc)
    std_accuracies.append(std_dev)
    
    mean_f1s.append(avg_f1)
    std_f1s.append(std_f1)
    
    mean_times.append(avg_time)
    
    print(f"Size {size} Avg Acc: {mean_accuracies[-1]:.2%} | Avg F1: {mean_f1s[-1]:.2f} | Time: {mean_times[-1]:.4f}s")

# %%
# 4. PLOTTING (Simple comparison)
plt.errorbar(TRAINING_SIZES, mean_accuracies, yerr=std_accuracies, fmt='-o', label='QSVC')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.title('Quantum SVM Performance (Simulated)')
plt.legend()
plt.grid(True)
plt.show()