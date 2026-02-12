"""
Data Manager Classes
==============================================
Author: Josiah Chan (K23091949)
Date: February 2026
Description: 
  This script benchmarks the performance and training time of a Quantum Support 
  Vector Machine (QSVC) against a Classical SVM (RBF Kernel).
  
  Key Features:
  - Data Diet Analysis: Testing performance across varying training set sizes.
  - Dimensionality Reduction: PCA pipeline to map 64-pixel images to N-qubits.

"""

# %%
# Imports
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

class BaseDataSplitter():
    """Handles train/test splitting logic - shared utility"""
    @staticmethod
    def create_imbalanced_split(X, y, train_size, seed, imbalance_ratio=0.5):
        """Create train/test split with class imbalance in training set"""
        X_class0 = X[y == 0]
        X_class1 = X[y == 1]
        
        n_class0 = int(train_size * imbalance_ratio)
        n_class1 = train_size - n_class0
        
        rng = np.random.default_rng(seed)
        idx0 = rng.choice(len(X_class0), size=min(n_class0, len(X_class0)), replace=False)
        idx1 = rng.choice(len(X_class1), size=min(n_class1, len(X_class1)), replace=False)
        
        X_train = np.vstack([X_class0[idx0], X_class1[idx1]])
        y_train = np.array([0] * len(idx0) + [1] * len(idx1))
        
        shuffle_idx = rng.permutation(len(y_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        return X_train, X_test, y_train, y_test


class DataManager():
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
        
        preprocessing_pipeline = Pipeline([
            ('pca', PCA(n_components=self.num_dims)),
            ('std_scaler', StandardScaler()),
            ('minmax_scaler', MinMaxScaler(feature_range=(-1, 1)))
        ])
        X_train_processed = preprocessing_pipeline.fit_transform(X_train)
        X_test_processed = preprocessing_pipeline.transform(X_test)
        
        return X_train_processed, X_test_processed, y_train, y_test


class AdhocDataManager():
    """Generates synthetic data"""
    def __init__(self, num_dims=4, num_samples=500, num_classes=2, 
                 class_sep=1.0, noise=0.1, random_state=42):
        self.num_dims = num_dims
        
        self.X, self.y = make_classification(
            n_samples=num_samples,
            n_features=num_dims,
            n_informative=max(2, num_dims - 2),
            n_redundant=min(2, num_dims - 2),
            n_classes=num_classes,
            class_sep=class_sep,
            flip_y=noise,
            random_state=random_state
        )
        
        print(f"Generated synthetic dataset: {num_samples} samples, {num_dims} features")
        
    def get_data_split(self, train_size, seed, imbalance_ratio=0.5):
        X_train, X_test, y_train, y_test = BaseDataSplitter.create_imbalanced_split(
            self.X, self.y, train_size, seed, imbalance_ratio
        )
        
        preprocessing_pipeline = Pipeline([
            ('std_scaler', StandardScaler()),
            ('minmax_scaler', MinMaxScaler(feature_range=(-1, 1)))
        ])
        X_train_processed = preprocessing_pipeline.fit_transform(X_train)
        X_test_processed = preprocessing_pipeline.transform(X_test)
        
        return X_train_processed, X_test_processed, y_train, y_test