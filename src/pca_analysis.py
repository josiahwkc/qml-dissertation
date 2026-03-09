"""
PCA Cumulative Explained Variance Analysis
========================================
Author: Josiah Chan (K23091949)

Description:
  Generates a publication-ready plot showing the cumulative explained variance 
  of PCA on the dataset. This is used to mathematically justify the choice of 
  N_DIM (number of qubits) for the quantum circuit by showing how much 
  information is retained vs. lost.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from data_manager import CSVDataManager

def plot_pca_variance(csv_filename, target_col, target_dimensions=4, max_samples=1000):
    """
    Loads dataset via CSVDataManager, applies PCA, and plots the explained variance.
    
    Args:
        csv_filename: Name of the dataset file.
        target_col: The target label column to exclude from PCA.
        target_dimensions: The N_DIM chosen for the QSVM (to highlight on graph).
        max_samples: Maximum samples to load (passed to CSVDataManager).
    """
    print(f"Analyzing PCA Variance for {csv_filename}...")
    
    # 1. Load Data using your existing data manager
    # This automatically handles missing values, categorical encoding, and subsampling
    manager = CSVDataManager(
        filename=csv_filename, 
        target_col=target_col,
        max_samples=max_samples
    )
    
    # Extract the cleaned, unscaled feature matrix
    X_raw = manager.X
    
    print(f"Extracted feature matrix shape: {X_raw.shape}")
    
    # 2. Scale Data (CRITICAL before PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 3. Fit PCA on ALL available dimensions to see the full spectrum
    pca = PCA()
    pca.fit(X_scaled)
    
    # Calculate variances
    exp_var = pca.explained_variance_ratio_ * 100
    cum_exp_var = np.cumsum(exp_var)
    
    # 4. Generate Publication-Ready Plot
    plt.figure(figsize=(10, 6))
    
    # Bar chart for individual variance
    plt.bar(range(1, len(exp_var) + 1), exp_var, alpha=0.5, align='center',
            label='Individual Explained Variance', color='steelblue')
            
    # Step plot for cumulative variance
    plt.step(range(1, len(cum_exp_var) + 1), cum_exp_var, where='mid',
             label='Cumulative Explained Variance', color='darkorange', linewidth=2)
    
    # Safety check: Ensure target_dimensions doesn't exceed available features
    plot_dim = min(target_dimensions, len(cum_exp_var))
    variance_at_target = cum_exp_var[plot_dim - 1]
    
    # Highlight the chosen N_DIM (e.g., 4 qubits)
    plt.axvline(x=plot_dim, color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=variance_at_target, color='red', linestyle='--', alpha=0.7)
    
    # Add an annotation dot and text
    plt.plot(plot_dim, variance_at_target, 'ro')
    plt.annotate(f'{variance_at_target:.1f}% variance\nat N={plot_dim}', 
                 xy=(plot_dim, variance_at_target), 
                 xytext=(plot_dim + 1, max(10, variance_at_target - 15)),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6))

    # Formatting
    plt.ylabel('Explained Variance Ratio (%)')
    plt.xlabel('Principal Component Index')
    plt.title(f'PCA Explained Variance: {csv_filename}')
    plt.xticks(range(1, len(exp_var) + 1, max(1, len(exp_var)//10))) # Make x-axis readable
    plt.ylim(0, 105)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Show and Save
    output_filename = f"pca_variance_{csv_filename.split('.')[0]}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Graph saved as {output_filename}")
    plt.show()


plot_pca_variance("breast-cancer.csv", "diagnosis", target_dimensions=5)