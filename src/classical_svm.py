# %%
# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
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
std_devs = []

for size in TRAINING_SIZES:
    trial_accuracies = []
    print(f"Training on subset size: {size} ... ", end="")

    for seed in range(N_TRIALS):
        # Step A: Sample a small subset from the Training Pool
        # stratify=y_train_pool ensures we get all digits (0-9) even in small sets
        X_subset, _, y_subset, _ = train_test_split(
            X_train_pool, y_train_pool, train_size=size, 
            random_state=seed, stratify=y_train_pool
        )

        # Step B: Train Classical SVM
        # SVC with RBF kernel is a very strong classical baseline
        clf = svm.SVC(kernel='rbf', gamma='scale', C=1.0)
        clf.fit(X_subset, y_subset)

        # Step C: Evaluate on the FIXED Test Set
        score = clf.score(X_test_fixed, y_test_fixed)
        trial_accuracies.append(score)

    # Step D: Average the results for this size
    avg_acc = np.mean(trial_accuracies)
    std_dev = np.std(trial_accuracies)
    
    mean_accuracies.append(avg_acc)
    std_devs.append(std_dev)
    
    print(f"Avg Accuracy: {avg_acc:.2%}")

# %%
# 4. VISUALIZATION

plt.figure(figsize=(8, 5))
plt.errorbar(TRAINING_SIZES, mean_accuracies, yerr=std_devs, 
                fmt='-o', capsize=5, label='Classical SVM')

plt.title('Classical SVM Performance vs. Data Availability')
plt.xlabel('Number of Training Samples')
plt.ylabel('Test Set Accuracy')
plt.grid(True)
plt.legend()
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
