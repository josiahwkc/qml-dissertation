import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_manager import CSVDataManager

def plot_pca_variance(csv_filename, target_col, max_samples=1000, limit=16, output_folder='pca analysis plots'):
    """
    Plots cumulative explained variance with an improved, high-clarity layout.
    """
    print(f"Analyzing PCA Variance for {csv_filename}...")
    
    # Assuming CSVDataManager is your custom loader
    manager = CSVDataManager()
    manager.load_dataset(filename=csv_filename, target_col=target_col, max_samples=max_samples)
    X_raw = manager.X
    
    # 1. Scale and Fit
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    pca = PCA()
    pca.fit(X_scaled)
    
    # 2. Extract Data
    exp_var_full = pca.explained_variance_ratio_ * 100
    cum_exp_var_full = np.cumsum(exp_var_full)
    
    plot_limit = min(limit, len(cum_exp_var_full))
    exp_var = exp_var_full[:plot_limit]
    cum_exp_var = cum_exp_var_full[:plot_limit]
    indices = np.arange(1, plot_limit + 1)
    
    # 3. Generate Plot
    plt.figure(figsize=(12, 7), facecolor='#fdfdfd') # Light clean background
    
    # Individual variance (Bars)
    bars = plt.bar(indices, exp_var, alpha=0.3, align='center',
                   label='Individual Variance', color='royalblue', edgecolor='navy')
            
    # Cumulative variance (Step + Area Fill)
    plt.step(indices, cum_exp_var, where='mid', label='Cumulative Variance', 
             color='darkorange', linewidth=2.5, zorder=3)

    # 4. Annotations and Markers
    for i, variance in enumerate(cum_exp_var):
        # Point markers
        plt.scatter(i + 1, variance, color='darkorange', s=30, zorder=4)
        
        # Labeling (offset text slightly based on variance value)
        plt.annotate(f'{variance:.1f}%', 
                     (i + 1, variance), 
                     textcoords="offset points", 
                     xytext=(0, 10), 
                     ha='center', 
                     fontsize=9, 
                     fontweight='bold',
                     color='black')

    # 6. Formatting & Aesthetics
    plt.ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Principal Components', fontsize=12, fontweight='bold')
    plt.title(f'PCA Information Retention: {csv_filename}\n(Top {plot_limit} Components)', 
              fontsize=14, pad=20)
    
    plt.xticks(indices)
    plt.xlim(0.5, plot_limit + 0.5)
    plt.ylim(0, 105) # Cap at 105% for label breathing room
    
    # Add minor gridlines for better readability
    plt.grid(axis='y', which='major', linestyle='-', alpha=0.2)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.legend(loc='lower right', frameon=True, shadow=True)
    plt.tight_layout()
    
    # Save and Show
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    output_folder = os.path.join(parent_dir, 'pca analysis plots')

    # Create the folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder at: {output_folder}")
        
    base_name = csv_filename.split('.')[0]
    file_name = f"pca_top_{plot_limit}_{base_name}.png"
    save_path = os.path.join(output_folder, file_name)
    
    plt.savefig(f"{save_path}.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Graph successfully saved to: {save_path}")