import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits # Use load_digits for a smaller version of MNIST
# OR use fetch_openml('mnist_784') for the full version (slower but standard)
from sklearn.metrics import accuracy_score

# 1. Load Data
print("Loading MNIST data...")
# We'll use the smaller 8x8 'digits' dataset for speed in prototyping.
# For the final dissertation, use fetch_openml('mnist_784')
digits = load_digits()
X = digits.data
y = digits.target

# 2. Filter for Binary Classification (0 vs 1)
# This makes it a binary problem, like breast cancer
binary_mask = (y == 0) | (y == 1)
X = X[binary_mask]
y = y[binary_mask]

print(f"Data shape after filtering: {X.shape}")

# 3. Preprocessing
# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce Dimensions (PCA)
# MNIST 8x8 has 64 features. We reduce to 10 for the quantum limit.
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X_scaled)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

# 4. Grid Search (Same as before)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

print("Starting Grid Search...")
svm = SVC(kernel='rbf')
grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Test Accuracy: {grid_search.score(X_test, y_test) * 100:.2f}%")

# 5. Visualize
scores = grid_search.cv_results_["mean_test_score"].reshape(
    len(param_grid['gamma']),
    len(param_grid['C'])
)

plt.figure(figsize=(8, 6))
sns.heatmap(scores,
            annot=True,
            xticklabels=param_grid['C'],     
            yticklabels=param_grid['gamma']) 
plt.xlabel('C Parameter')
plt.ylabel('Gamma Parameter')
plt.title("SVM Accuracy on Binary MNIST (0 vs 1)")
plt.show()