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
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from qiskit_machine_learning.datasets import ad_hoc_data

class BaseDataSplitter():
    """Handles train/test splitting logic - shared utility"""
        
    @staticmethod
    def create_class_imbalance(X_pool, y_pool, train_size, seed, imbalance_ratio=0.5):
        num_classes = len(np.unique(y_pool))
        rng = np.random.default_rng(seed)
        
        if num_classes == 2:
            # Binary classification - apply imbalance
            X_pool_class0 = X_pool[y_pool == 0]
            X_pool_class1 = X_pool[y_pool == 1]
            
            n_class0 = int(train_size * imbalance_ratio)
            n_class1 = train_size - n_class0
            
            idx0 = rng.choice(len(X_pool_class0), size=min(n_class0, len(X_pool_class0)), replace=False)
            idx1 = rng.choice(len(X_pool_class1), size=min(n_class1, len(X_pool_class1)), replace=False)
            
            X_train = np.vstack([X_pool_class0[idx0], X_pool_class1[idx1]])
            y_train = np.array([0] * len(idx0) + [1] * len(idx1))
            
        else:
            # Multi-class - balanced sampling across all classes            
            samples_per_class = train_size // num_classes
            remainder = train_size % num_classes
            
            X_train_list = []
            y_train_list = []
            
            for class_label in range(num_classes):
                X_class = X_pool[y_pool == class_label]
                
                # Distribute remainder across first few classes
                n_samples = samples_per_class + (1 if class_label < remainder else 0)
                n_samples = min(n_samples, len(X_class))
                
                idx = rng.choice(len(X_class), size=n_samples, replace=False)
                X_train_list.append(X_class[idx])
                y_train_list.append(np.full(len(idx), class_label))
            
            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)
        
        # Shuffle
        shuffle_idx = rng.permutation(len(y_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        
        return X_train, y_train


class AdhocDataManager():
    """Generates synthetic data using Qiskit's ad_hoc_data"""
    def __init__(self, num_dims=2, gap=0.3):
        self.num_dims = num_dims
                
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
        
        self.X_test_fixed = X_test_fixed
        self.y_test_fixed = np.argmax(y_test_fixed, axis=1)
        
    def get_data_split(self, train_size, seed, imbalance_ratio=0.5):
        X_train, y_train = BaseDataSplitter.create_class_imbalance(
            self.X_pool, self.y_pool, train_size, seed, imbalance_ratio
        )
        
        return X_train, self.X_test_fixed, y_train, self.y_test_fixed


class CSVDataManager():
    """Manages arbitrary CSV datasets with automatic preprocessing"""
    def __init__(self, filename, target_col, num_dims=4, n_class=None, 
                 categorical_cols=None, drop_cols=None, max_samples=1000):
        """
        Args:
            filename: CSV filename in the datasets/ folder
            target_col: Name of the target column to predict
            num_dims: Number of PCA components for dimensionality reduction
            n_class: Number of classes to keep (None = keep all)
            categorical_cols: List of categorical column names to encode
            drop_cols: List of columns to drop before processing
            max_samples: Maximum total samples to use (for large datasets like MNIST)
        """
        self.num_dims = num_dims
        self.target_col = target_col
        
        # Locate the CSV file
        current_script = Path(__file__).resolve()
        project_root = current_script.parent.parent
        file_path = project_root / "datasets" / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")
            
        print(f"Loading data from: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Original shape: {df.shape}")
        
        # Drop specified columns
        if drop_cols:
            df = df.drop(columns=drop_cols, errors='ignore')
            print(f"After dropping columns: {df.shape}")
        
        # Separate features and target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found. Available: {df.columns.tolist()}")
        
        y_raw = df[target_col].values
        X_df = df.drop(columns=[target_col])
        
        # Handle categorical features
        if categorical_cols is None:
            categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if categorical_cols:
            print(f"Encoding categorical columns: {categorical_cols}")
            for col in categorical_cols:
                if col in X_df.columns:
                    le = LabelEncoder()
                    X_df[col] = le.fit_transform(X_df[col].astype(str))
        
        # Handle missing values
        if X_df.isnull().any().any():
            print("Warning: Missing values detected. Filling with column means.")
            X_df = X_df.fillna(X_df.mean())
        
        # Convert to numpy
        X_raw = X_df.values
        
        # Encode target labels
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(y_raw)
        self.target_classes = le_target.classes_
        print(f"Target classes: {self.target_classes}")
        
        # Filter to n_class if specified
        if n_class is not None:
            unique_classes = np.unique(y_encoded)
            if n_class > len(unique_classes):
                print(f"Warning: Requested {n_class} classes but only {len(unique_classes)} available")
                n_class = len(unique_classes)
            
            mask = y_encoded < n_class
            X_filtered = X_raw[mask]
            y_filtered = y_encoded[mask]
            print(f"Filtered to {n_class} classes: {len(y_filtered)} samples")
        else:
            X_filtered = X_raw
            y_filtered = y_encoded
            print(f"Using all {len(np.unique(y_filtered))} classes: {len(y_filtered)} samples")
        
        if len(y_filtered) > max_samples:
            # Stratified sampling to keep class balance
            X_sampled, _, y_sampled, _  = train_test_split(
                X_filtered, y_filtered,
                train_size=max_samples,
                random_state=42,
                stratify=y_filtered
            )
            self.X = X_sampled
            self.y = y_sampled
            print(f"Subsampled to {len(self.y)} samples (stratified)")
        else:
            self.X = X_filtered
            self.y = y_filtered
        
        # Verify we have enough features for PCA
        if self.X.shape[1] < num_dims:
            print(f"Warning: Only {self.X.shape[1]} features available, reducing num_dims to {self.X.shape[1]}")
            self.num_dims = self.X.shape[1]

    def get_data_split(self, train_size, seed, imbalance_ratio=0.5):
        """Create train/test split with preprocessing"""
        # Split into pool and test
        X_pool, X_test, y_pool, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=seed, stratify=self.y
        )
        
        # Preprocessing pipeline
        preprocessing_pipeline = Pipeline([
            ('pca', PCA(n_components=self.num_dims)),
            ('std_scaler', StandardScaler()), 
            ('minmax_scaler', MinMaxScaler(feature_range=(-1, 1)))
        ])
        
        # Fit on pool, transform both
        X_pool_processed = preprocessing_pipeline.fit_transform(X_pool)
        X_test_processed = preprocessing_pipeline.transform(X_test)
        
        # Sample training subset
        X_train_processed, y_train = BaseDataSplitter.create_class_imbalance(
            X_pool_processed, y_pool, train_size, seed, imbalance_ratio
        )
        
        return X_train_processed, X_test_processed, y_train, y_test