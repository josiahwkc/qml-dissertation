# %%
# IMPORTS
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# %%
# 1. CONFIGURATION
# The sizes of the training sets we want to test (The "Data Diet")
TRAINING_SIZES = [50, 100, 300, 500, 1000]

# How many times to repeat each size (to average out luck)
N_TRIALS = 5

N_COMPONENTS = 16

# %%
# 2. DATA PREPARATION
print("Loading digits dataset...")
# Loading standard MNIST (8x8 version built into sklearn for speed)
# If you use the full 28x28 MNIST, just replace this block to load that instead.
digits = datasets.load_digits()
X = digits.data
y = digits.target

# SCALING (Crucial for SVM and Quantum)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dimensionality Reduction
print(f"Reducing dimensions to {N_COMPONENTS} using PCA...")
pca = PCA(n_components=N_COMPONENTS)
X = pca.fit_transform(X)

# CREATE A FIXED TEST SET
# We hold out 20% of data for testing. This set is NEVER used for training.
X_train_pool, X_test_fixed, y_train_pool, y_test_fixed = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Fixed Test Set Size: {len(X_test_fixed)} samples")
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
    
    print(f"Training on subset size: {size} ... ")

    for seed in range(N_TRIALS):
        # Step A: Sample a small subset from the Training Pool
        # stratify=y_train_pool ensures we get all digits (0-9) even in small sets
        X_subset, _, y_subset, _ = train_test_split(
            X_train_pool, y_train_pool, train_size=size, 
            random_state=seed, stratify=y_train_pool
        )
        
        #Step B: Grid Search for Optimal Hyperparameters
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [0.001, 0.01, 0.1, 1]
        }
        
        grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_subset, y_subset)
        print(f"Best Parameters: {grid_search.best_params_}")

        best_C = grid_search.best_params_['C']
        best_gamma = grid_search.best_params_['gamma']
        
        # Step C: Train Classical SVM using extracted hyperparameters
        # SVC with RBF kernel is a very strong classical baseline
        clf = svm.SVC(kernel='rbf', gamma=best_gamma, C=best_C)
        
        start_time = time.time()
        clf.fit(X_subset, y_subset)
        end_time = time.time()

        # Step D: Evaluate on the FIXED Test Set
        trial_times.append(end_time - start_time)
        
        y_pred = clf.predict(X_test_fixed)
        
        score = metrics.accuracy_score(y_test_fixed, y_pred)
        trial_accuracies.append(score)
        
        f1 = f1_score(y_test_fixed, y_pred, average='weighted')
        trial_f1s.append(f1)

    # Step E: Average the results for this size
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
    
    print(f"Avg Acc: {mean_accuracies[-1]:.2%} | Avg F1: {mean_f1s[-1]:.2f} | Time: {mean_times[-1]:.4f}s")

# %%
# 4. VISUALIZATION

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Accuracy
ax1.errorbar(TRAINING_SIZES, mean_accuracies, yerr=std_accuracies, 
             fmt='-o', capsize=5, label='Classical SVM', color='blue')
ax1.set_title('Test Set Accuracy')
ax1.set_xlabel('Training Samples')
ax1.set_ylabel('Accuracy')
ax1.grid(True)
ax1.legend()

# Plot 2: F1 Score (The new metric)
ax2.errorbar(TRAINING_SIZES, mean_f1s, yerr=std_f1s, 
             fmt='-s', capsize=5, label='Classical SVM', color='green')
ax2.set_title('Test Set F1-Score (Weighted)')
ax2.set_xlabel('Training Samples')
ax2.set_ylabel('F1 Score')
ax2.set_ylim(0, 1) # F1 is always between 0 and 1
ax2.grid(True)
ax2.legend()

# Plot 3: Training Time
ax3.plot(TRAINING_SIZES, mean_times, '-o', label='Classical SVM', color='red')
ax3.set_title('Training Time vs Data Size')
ax3.set_xlabel('Training Samples')
ax3.set_ylabel('Time (Seconds)')
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.show()

# %%
# 5. VISUALIZATION OF DECISION BOUNDARIES
# Run this AFTER running Cell 2 (Data Prep).
# This creates a "Map" of how the model sees the data.

print("Generating 2D Visual Map...")

# RETRAIN A SINGLE MODEL (To ensure we have a good one to look at)
# We use a large training size (1000) to get a clear picture
X_vis_train, _, y_vis_train, _ = train_test_split(
    X_train_pool, y_train_pool, train_size=1000, 
    random_state=42, stratify=y_train_pool
)

# Train the "Brain" (The 16-Feature Model)
# This is the model that actually decides if a digit is a '0' or '1'
model_for_vis = svm.SVC(kernel='rbf', gamma='scale', C=1.0)
model_for_vis.fit(X_vis_train, y_vis_train)

# CREATE THE 2D PROJECTION (The "Map")
# We take the Test Set (which is 16D) and squash it to 2D
# Note: If X is already 16D, this finds the top 2 dimensions within those 16.
pca_2d = PCA(n_components=2)
X_test_2D = pca_2d.fit_transform(X_test_fixed)

# GET PREDICTIONS
# The model looks at the 16D data to make its guess
predictions = model_for_vis.predict(X_test_fixed)

# PLOT
plt.figure(figsize=(12, 8))

# Scatter plot: 
# x and y positions come from the 2D "Map"
# Colors (c) come from the 16D "Brain" predictions
scatter = plt.scatter(X_test_2D[:, 0], X_test_2D[:, 1], 
                      c=predictions, cmap='tab10', alpha=0.6, s=20)

# Add a legend
plt.colorbar(scatter, label='Digit Predicted by Model')
plt.title(f"Model Decisions Visualized (Positions=2D PCA, Colors={N_COMPONENTS}D Prediction)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True, alpha=0.3)
plt.show()
# %%
