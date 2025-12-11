import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_classical_benchmark():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    # The sizes of the training sets we want to test (The "Data Diet")
    training_sizes = [50, 100, 300, 500, 1000] 
    
    # How many times to repeat each size (to average out luck)
    n_trials = 5 
    
    # Set to True to mimic QML inputs (e.g., 16 features instead of 64/784)
    USE_PCA = True 
    N_COMPONENTS = 16 

    # ==========================================
    # 2. DATA PREPARATION
    # ==========================================
    print("Loading digits dataset...")
    # Loading standard MNIST (8x8 version built into sklearn for speed)
    # If you use the full 28x28 MNIST, just replace this block to load that instead.
    digits = datasets.load_digits()
    X = digits.data
    y = digits.target

    # SCALING (Crucial for SVM and Quantum)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # OPTIONAL: PCA (Dimensionality Reduction)
    if USE_PCA:
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

    # ==========================================
    # 3. THE EXPERIMENT LOOP
    # ==========================================
    mean_accuracies = []
    std_devs = []

    for size in training_sizes:
        trial_accuracies = []
        print(f"Training on subset size: {size} ... ", end="")

        for seed in range(n_trials):
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

    # ==========================================
    # 4. VISUALIZATION
    # ==========================================
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(training_sizes, mean_accuracies, yerr=std_devs, 
                 fmt='-o', capsize=5, label='Classical SVM')
    
    plt.title('Classical SVM Performance vs. Data Availability')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Test Set Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Return results if you want to use them later
    return training_sizes, mean_accuracies

if __name__ == "__main__":
    run_classical_benchmark()