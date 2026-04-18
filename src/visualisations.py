import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_manager import CSVDataManager

def plot_dataset_comparison(datasets_dict, title="Qualitative Dataset Comparison (2D PCA Projection)"):
    """
    Plots a side-by-side comparison of multiple datasets using their first 2 PCA components.
    
    Args:
        datasets_dict (dict): A dictionary where keys are dataset names (str) 
                              and values are tuples of (X, y) arrays.
                              Example: {"Cancer Data": (X1, y1), "Bank Data": (X2, y2)}
        title (str): Main title for the figure.
    """
    num_datasets = len(datasets_dict)
    
    # Dynamically calculate grid size (e.g., 3 datasets = 1x3, 4 datasets = 2x2)
    cols = min(3, num_datasets)
    rows = int(np.ceil(num_datasets / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    
    # Flatten axes array for easy iteration, even if it's 1D or 2D
    if num_datasets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    sns.set_theme(style="whitegrid", palette="deep")
    
    for ax, (name, (X, y)) in zip(axes, datasets_dict.items()):
        # 1. Standardize and reduce to 2D for visualization
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # 2. Calculate variance explained by these top 2 components
        var_explained = sum(pca.explained_variance_ratio_) * 100
        
        # 3. Create DataFrame for Seaborn
        df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        df['Class'] = [f"Class {label}" for label in y]
        
        # 4. Plot the scatter
        sns.scatterplot(
            data=df, x="PC1", y="PC2", hue="Class", 
            ax=ax, alpha=0.7, edgecolor='w', s=50
        )
        
        # 5. Clean up formatting
        ax.set_title(f"PCA Projection of {name}\n(First Two Components)", fontweight='bold')
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        
        # Simplify legends to avoid clutter
        if ax.get_legend() is not None:
            ax.legend(title="", loc="best", framealpha=0.8)

    # Hide any unused subplots if the grid is larger than the number of datasets
    for i in range(num_datasets, len(axes)):
        fig.delaxes(axes[i])
        
    fig.suptitle(title, y=1.05, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
# Load your datasets
dm1 = CSVDataManager()
dm1.load_dataset(filename="breast-cancer.csv", target_col="diagnosis")

dm2 = CSVDataManager()
dm2.load_dataset(filename="mnist_train.csv", target_col="label")

dm3 = CSVDataManager()
dm3.load_dataset(filename="fashion-mnist_train.csv", target_col="label")

# Create the dictionary
my_datasets = {
    "Breast Cancer": (dm1.X, dm1.y),
    "MNIST": (dm2.X, dm2.y),
    "Fashion-MNIST": (dm3.X, dm3.y)
}

# Generate the comparison graphic
plot_dataset_comparison(my_datasets)