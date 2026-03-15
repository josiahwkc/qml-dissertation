import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Assuming CSVDataManager is imported as before
from data_manager import CSVDataManager

def plot_pca_variance(csv_filename, target_col, max_samples=1000, limit=16):
    """
    Plots cumulative explained variance for the first 'limit' components.
    """
    print(f"Analyzing PCA Variance for {csv_filename} (Max {limit} components)...")
    
    manager = CSVDataManager(filename=csv_filename, target_col=target_col, max_samples=max_samples)
    X_raw = manager.X
    
    # 1. Scale and Fit
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    # 2. Extract and Slice to the first 16 components
    exp_var_full = pca.explained_variance_ratio_ * 100
    cum_exp_var_full = np.cumsum(exp_var_full)
    
    # Slice arrays to the user-defined limit
    plot_limit = min(limit, len(cum_exp_var_full))
    exp_var = exp_var_full[:plot_limit]
    cum_exp_var = cum_exp_var_full[:plot_limit]
    indices = range(1, plot_limit + 1)
    
    # 3. Generate Plot
    plt.figure(figsize=(12, 6))
    
    # Individual variance bars
    plt.bar(indices, exp_var, alpha=0.4, align='center',
            label='Individual Explained Variance', color='steelblue')
            
    # Cumulative variance step plot
    plt.step(indices, cum_exp_var, where='mid',
             label='Cumulative Explained Variance', color='darkorange', linewidth=2)

    # 4. Label every N (1 through 16)
    for i, variance in enumerate(cum_exp_var):
        plt.text(i + 1, variance + 1.5, f'{variance:.1f}%', 
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Point marker at each step
        plt.plot(i + 1, variance, 'o', color='darkorange', markersize=4)

    # 5. Formatting
    plt.ylabel('Explained Variance Ratio (%)')
    plt.xlabel('Principal Component Index')
    plt.title(f'Top {plot_limit} PCA Components: {csv_filename}')
    
    plt.xticks(indices) # Force show all 16 indices
    plt.ylim(0, 110)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    # Save and Show
    output_filename = f"pca_top_{plot_limit}_{csv_filename.split('.')[0]}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Graph saved as {output_filename}")
    plt.show()

# Run it
plot_pca_variance("mnist_train.csv", "label", limit=16)