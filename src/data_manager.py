"""
Data Manager Classes
====================
Author: Josiah Chan (K23091949)

Description: 
  Provides data loading and preprocessing utilities for quantum and classical 
  SVM experiments. Supports both real-world datasets (MNIST digits) and 
  synthetic datasets with configurable properties.
  
Key Classes:
  - DataManager: Loads MNIST digits with PCA dimensionality reduction
  - AdhocDataManager: Generates synthetic data using Qiskit's ad_hoc_data
  
Key Features:
  - Configurable class imbalance ratios for training sets
  - Balanced test sets for fair evaluation
  - Preprocessing pipelines (PCA, StandardScaler, MinMaxScaler)
  - Reproducible splits via random seeds
"""

# %%
# Imports
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from qiskit_machine_learning.datasets import ad_hoc_data

class BaseDataSplitter():
    """Handles train/test splitting logic - shared utility"""
    @staticmethod
    def create_imbalanced_split(X, y, train_size, seed, imbalance_ratio=0.5):
        """
        Create train/test split with class imbalance in training set
        
        Args:
            X: Full feature matrix
            y: Full label vector
            train_size: Number of samples in training set
            seed: Random seed for reproducibility
            imbalance_ratio: Proportion of class 0 in training set (0.5 = balanced)
        
        Returns:
            X_train, X_test, y_train, y_test (test set is always balanced)
        """
        X_pool, X_test, y_pool, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_pool_class0 = X_pool[y_pool == 0]
        X_pool_class1 = X_pool[y_pool == 1]
        
        n_class0 = int(train_size * imbalance_ratio)
        n_class1 = train_size - n_class0
        
        rng = np.random.default_rng(seed)
        idx0 = rng.choice(len(X_pool_class0), size=min(n_class0, len(X_pool_class0)), replace=False)
        idx1 = rng.choice(len(X_pool_class1), size=min(n_class1, len(X_pool_class1)), replace=False)
        
        X_train = np.vstack([X_pool_class0[idx0], X_pool_class1[idx1]])
        y_train = np.array([0] * len(idx0) + [1] * len(idx1))
        
        shuffle_idx = rng.permutation(len(y_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        
        return X_train, X_test, y_train, y_test


class MNISTDataManager():
    """Manages real MNIST digit data"""
    def __init__(self, num_dims=4, n_class=2):
        self.num_dims = num_dims
        digits = datasets.load_digits(n_class=n_class)
        self.X = digits.data
        self.y = digits.target
        
    def get_data_split(self, train_size, seed, imbalance_ratio=0.5):
        X_train, X_test, y_train, y_test = BaseDataSplitter.create_imbalanced_split(
            self.X, self.y, train_size, seed, imbalance_ratio
        )
        
        train_indices = set([tuple(x) for x in X_train])
        test_indices = set([tuple(x) for x in X_test])
        overlap = train_indices & test_indices
        print(f"Samples in both train and test: {len(overlap)}")  # Should be 0
        
        preprocessing_pipeline = Pipeline([
            ('pca', PCA(n_components=self.num_dims)),
            ('std_scaler', StandardScaler()),
            ('minmax_scaler', MinMaxScaler(feature_range=(-1, 1)))
        ])
        X_train_processed = preprocessing_pipeline.fit_transform(X_train)
        X_test_processed = preprocessing_pipeline.transform(X_test)
        
        return X_train_processed, X_test_processed, y_train, y_test


class AdhocDataManager():
    """Generates synthetic data using Qiskit's ad_hoc_data"""
    def __init__(self, num_dims=2, gap=0.3):
        self.num_dims = num_dims
        
        print(f"Generating Qiskit Ad Hoc Data (Dims={num_dims}, Gap={gap})...")
        
        # Generate fixed pools
        X_train_pool, y_train_pool, X_test_fixed, y_test_fixed = ad_hoc_data(
            training_size=600, 
            test_size=200, 
            n=num_dims, 
            gap=gap, 
            plot_data=False
        )
        
        # Convert one-hot to 1D labels
        self.X_pool = X_train_pool
        self.y_pool = np.argmax(y_train_pool, axis=1)
        
        self.X_test_fixed = X_test_fixed  # Never changes
        self.y_test_fixed = np.argmax(y_test_fixed, axis=1)
        
    def get_data_split(self, train_size, seed, imbalance_ratio=0.5):
        """Sample from training pool, always use fixed test set"""
        # Separate classes in training pool
        X_pool_class0 = self.X_pool[self.y_pool == 0]
        X_pool_class1 = self.X_pool[self.y_pool == 1]
        
        # Calculate samples per class
        n_class0 = int(train_size * imbalance_ratio)
        n_class1 = train_size - n_class0
        
        # Sample from pools
        rng = np.random.default_rng(seed)
        idx0 = rng.choice(len(X_pool_class0), size=min(n_class0, len(X_pool_class0)), replace=False)
        idx1 = rng.choice(len(X_pool_class1), size=min(n_class1, len(X_pool_class1)), replace=False)
        
        X_train = np.vstack([X_pool_class0[idx0], X_pool_class1[idx1]])
        y_train = np.array([0] * len(idx0) + [1] * len(idx1))
        
        # Shuffle
        shuffle_idx = rng.permutation(len(y_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        
        # Return with FIXED test set
        return X_train, self.X_test_fixed, y_train, self.y_test_fixed