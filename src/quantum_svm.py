import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Quantum Imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# CONFIGURATION
N_DIM = 4 
TRAIN_SIZE = 100
TRAINING_SIZES = [20, 40, 60, 80, 100]
N_TRIALS = 3
TEST_SIZE = 20
N_CLASS = 2

digits = datasets.load_digits(n_class=N_CLASS)

X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=22
)


preprocessing_pipeline = Pipeline([
    ('pca', PCA(n_components=N_DIM)),             
    ('std_scaler', StandardScaler()),             
    ('minmax_scaler', MinMaxScaler(feature_range=(-1, 1)))
])

X_train_processed = preprocessing_pipeline.fit_transform(X_train)
X_test_processed = preprocessing_pipeline.transform(X_test)

print(f"Original Feature Count: {X_train.shape[1]}")
print(f"Processed Feature Count: {X_train_processed.shape[1]} (Should match N_DIM={N_DIM})")

# Limit dataset size for simulation speed
# We slice AFTER processing to ensure the pipeline is fit on the full distribution
X_train_processed = X_train_processed[:TRAIN_SIZE]
y_train = y_train[:TRAIN_SIZE]
X_test_processed = X_test_processed[:TEST_SIZE]
y_test = y_test[:TEST_SIZE]

print(f"Final Dataset: {len(X_train_processed)} training samples, {N_DIM} features.")

# DEFINE QUANTUM KERNEL
feature_map = ZZFeatureMap(feature_dimension=N_DIM, reps=2, entanglement='linear')
kernel = FidelityQuantumKernel(feature_map=feature_map)

# PRECOMPUTE KERNEL MATRICES
print("Calculating Training Kernel Matrix (Simulating Quantum Circuit)...")
start_q_time = time.time()

try:
    matrix_train = kernel.evaluate(x_vec=X_train_processed)
    matrix_test = kernel.evaluate(x_vec=X_test_processed, y_vec=X_train_processed)
except ValueError as e:
    print(f"\nCRITICAL ERROR: {e}")
    print("This usually means N_DIM is too high or PCA failed.")
    exit()

end_q_time = time.time()
quantum_time = end_q_time - start_q_time
print(f"Quantum Kernel Calculation Time: {quantum_time:.2f} seconds")

# VISUALIZE KERNEL MATRIX
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(matrix_train, interpolation='nearest', origin='upper', cmap='Blues')
plt.title("Training Kernel Matrix")
plt.subplot(1, 2, 2)
plt.imshow(matrix_test, interpolation='nearest', origin='upper', cmap='Reds')
plt.title("Testing Kernel Matrix")
plt.show()

# TRAIN SVM
print("Training Classical SVM on Quantum Kernel...")
start_c_time = time.time()

qsvm = svm.SVC(kernel='precomputed', C=1.0)
qsvm.fit(matrix_train, y_train)

end_c_time = time.time()
classical_time = end_c_time - start_c_time
print(f"SVM Training Time: {classical_time:.4f} seconds")

# EVALUATE
score = qsvm.score(matrix_test, y_test)
print(f"\nFinal Test Accuracy: {score:.2%}")