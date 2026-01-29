# %%
# IMPORTS
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from grid_search import GridSearch

# %%
# 1. CONFIGURATION
TRAINING_SIZES = [50, 100, 300, 500, 1000]
N_TRIALS = 5
N_COMPONENTS = 16

# %%
# 2. DATA PREPARATION
#print("Loading digits dataset...")
digits = datasets.load_digits()
X_raw = digits.data
y = digits.target

# Split raw data
X_train_pool, X_test_fixed_raw, y_train_pool, y_test_fixed = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train Pool Size: {len(X_train_pool)} samples")
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
    
    print(f"Training on subset size: {size}")

    for seed in range(N_TRIALS):
        # Sample a small subset from the Training Pool
        # stratify=y_train_pool ensures we get all digits (0-9) even in small sets
        X_subset_raw, _, y_subset, _ = train_test_split(
            X_train_pool, y_train_pool, train_size=size, 
            random_state=seed, stratify=y_train_pool
        )
        
        preprocessor = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=N_COMPONENTS))
        ])
        
        preprocessor.fit(X_subset_raw)
        
        X_subset_transformed = preprocessor.transform(X_subset_raw)
        X_test_transformed = preprocessor.transform(X_test_fixed_raw)
        
        # ====================================
        # COMMENTED OUT HYPERPARAMETER TUNING
        # ====================================
        # best_C, best_gamma = GridSearch.run(X_subset_transformed, y_subset)
        
        # Train Classical SVM using extracted hyperparameters
        clf = svm.SVC(kernel='rbf')
        
        start_time = time.time()
        clf.fit(X_subset_transformed, y_subset)
        end_time = time.time()

        # Evaluate on the FIXED Test Set
        trial_times.append(end_time - start_time)
        
        y_pred = clf.predict(X_test_transformed)
        
        score = metrics.accuracy_score(y_test_fixed, y_pred)
        trial_accuracies.append(score)
        
        f1 = f1_score(y_test_fixed, y_pred, average='weighted')
        trial_f1s.append(f1)

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

# Plot Accuracy
ax1.errorbar(TRAINING_SIZES, mean_accuracies, yerr=std_accuracies, 
             fmt='-o', capsize=5, label='Classical SVM', color='blue')
ax1.set_title('Test Set Accuracy')
ax1.set_xlabel('Training Samples')
ax1.set_ylabel('Accuracy')
ax1.grid(True)
ax1.legend()

# Plot F1 Score
ax2.errorbar(TRAINING_SIZES, mean_f1s, yerr=std_f1s, 
             fmt='-s', capsize=5, label='Classical SVM', color='green')
ax2.set_title('Test Set F1-Score (Weighted)')
ax2.set_xlabel('Training Samples')
ax2.set_ylabel('F1 Score')
ax2.set_ylim(0, 1)
ax2.grid(True)
ax2.legend()

# Plot Training Time
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

print("Generating 2D Visual Map...")

# 1. Sample Training Data
X_vis_train_raw, _, y_vis_train, _ = train_test_split(
    X_train_pool, y_train_pool, train_size=1000, 
    random_state=42, stratify=y_train_pool
)

# 2. Fit Preprocessing on THIS set only
vis_preprocessor = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=N_COMPONENTS))
])
vis_preprocessor.fit(X_vis_train_raw)

X_vis_train_pca = vis_preprocessor.transform(X_vis_train_raw)
X_test_fixed_pca = vis_preprocessor.transform(X_test_fixed_raw) # Transform Test set

# 3. Train Model
model_for_vis = svm.SVC(kernel='rbf', gamma='scale', C=1.0)
model_for_vis.fit(X_vis_train_pca, y_vis_train)

# 4. Create 2D Projection for Plotting (Fit on TRAIN, Apply to TEST)
pca_2d = PCA(n_components=2)
pca_2d.fit(X_vis_train_pca) 
X_test_2D = pca_2d.transform(X_test_fixed_pca)

# 5. Predict
predictions = model_for_vis.predict(X_test_fixed_pca)

# 6. Plot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_test_2D[:, 0], X_test_2D[:, 1], 
                      c=predictions, cmap='tab10', alpha=0.6, s=20)
plt.colorbar(scatter, label='Digit Predicted by Model')
plt.title(f"Model Decisions (Preprocessing Fitted on Train Only)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True, alpha=0.3)
plt.show()

# %%
