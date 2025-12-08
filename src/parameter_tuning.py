import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

# 1. Load and Preprocess Data
data = load_breast_cancer()
X = data.data
y = data.target

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce Dimensions (PCA) - Crucial for fair comparison with Quantum
# We reduce to 10 features to simulate the 10-qubit limit
pca = PCA(n_components=10)
X_reduced = pca.fit_transform(X_scaled)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

# 2. Define Parameter Grid for RBF Kernel
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [0.001, 0.01, 0.1, 1, 10]
}

# 3. Perform Grid Search
print("Starting Grid Search...")
svm = SVC(kernel='rbf')
grid_search = GridSearchCV(
    svm, 
    param_grid, 
    cv=5,       # 5-fold cross-validation is standard
    n_jobs=-1,  # Use all CPU cores
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Val Score: {grid_search.best_score_ * 100:.2f}%")

# 4. Evaluate on Test Set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {test_acc * 100:.2f}%")

# 5. Visualizations (Heatmap)
# Reshape results for heatmap
scores = grid_search.cv_results_["mean_test_score"].reshape(
    len(param_grid['gamma']),
    len(param_grid['C'])
)

plt.figure(figsize=(8, 6))
sns.heatmap(scores, annot=True, cmap='viridis', fmt=".3f")
plt.xlabel("C Parameter")
plt.ylabel("Gamma Parameter")
plt.xticks(np.arange(len(param_grid['C'])) + 0.5, param_grid['C'])
plt.yticks(np.arange(len(param_grid['gamma'])) + 0.5, param_grid['gamma'])
plt.title("SVM Accuracy Heatmap (PCA=10)")
plt.show()