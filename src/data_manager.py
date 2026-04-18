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


import pandas as pd
from pathlib import Path
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from qiskit_machine_learning.datasets import ad_hoc_data

class TrainingSampler():
    """Samples training subsets with optional class imbalance"""
        
    @staticmethod
    def create_class_imbalance(X_pool, y_pool, train_size, seed, imbalance_ratio=0.5):
        rng = np.random.default_rng(seed)
        
        X_pool_class0 = X_pool[y_pool == 0]
        X_pool_class1 = X_pool[y_pool == 1]
        
        n_class0 = int(train_size * imbalance_ratio)
        n_class1 = train_size - n_class0
        
        idx0 = rng.choice(len(X_pool_class0), size=min(n_class0, len(X_pool_class0)), replace=False)
        idx1 = rng.choice(len(X_pool_class1), size=min(n_class1, len(X_pool_class1)), replace=False)
        
        X_train = np.vstack([X_pool_class0[idx0], X_pool_class1[idx1]])
        y_train = np.array([0] * len(idx0) + [1] * len(idx1))
        
        # Shuffle
        shuffle_idx = rng.permutation(len(y_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        
        return X_train, y_train


class QuantumBenchmarkDataManager():
    """Generates synthetic data using Qiskit's ad_hoc_data"""
    
    def __init__(self):
        self.X_pool = None
        self.y_pool = None
        self.X_test_fixed = None
        self.y_test_fixed = None
    
    def create_dataset(self, num_dims=2, gap=0.3, pool_size=800, test_size=200):
        """
        Generate synthetic data using ad_hoc_data.
        
        Args:
            num_dims: Number of features
            gap: Class separation
            pool_size: Size of training pool for sampling
            test_size: Size of fixed test set
        """
        per_class_train = pool_size // 2
        per_class_test = test_size // 2
        
        # Generate data pool + fixed test set
        X_pool, y_pool, X_test, y_test = ad_hoc_data(
            training_size=per_class_train,
            test_size=per_class_test,
            n=num_dims,
            gap=gap,
            plot_data=True
        )
        
        # Convert one-hot to 1D labels
        self.X_pool = X_pool
        self.y_pool = np.argmax(y_pool, axis=1)
        self.X_test_fixed = X_test
        self.y_test_fixed = np.argmax(y_test, axis=1)
    
    def get_data_split(self, seed):
        """Create train/test/validation split"""
        
        # Split into 80% (of 800) train and 20% (of 200) validation
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_pool, self.y_pool, test_size=0.2, random_state=seed, stratify=self.y_pool
        )
        
        return X_train, X_val, self.X_test_fixed, y_train, y_val, self.y_test_fixed


class CSVDataManager():
    """Manages arbitrary CSV datasets with automatic preprocessing"""
    def __init__(self):
        self.filename = None
        self.X = None
        self.y = None
    
    def load_dataset(self, filename, target_col, num_dims=5, n_class=2, 
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
        self.filename = filename
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
    
    def preprocess_data(self, X_train, X_val, X_test):
        preprocessing_pipeline = Pipeline([
            ('std_scaler', StandardScaler()),
            ('pca', PCA(n_components=self.num_dims)),
            ('minmax_scaler', MinMaxScaler(feature_range=(0, 1)))
        ])
        
        # Fit on X_train, transform all
        X_train_processed = preprocessing_pipeline.fit_transform(X_train)
        X_test_processed = preprocessing_pipeline.transform(X_test)
        X_val_processed = preprocessing_pipeline.transform(X_val)
        
        return X_train_processed, X_val_processed, X_test_processed
    
    def get_kfold_splits(self, seed, k_folds=5):
        """
        Generator that yields k mutually exclusive splits of the data.
        """
        # Split into pool and test
        X_pool, X_test, y_pool, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=seed, stratify=self.y
        )
        
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        
        # Split X_pool into training and validation folds
        for train_idx, val_idx in skf.split(X_pool, y_pool):
            X_train, X_val = X_pool[train_idx], X_pool[val_idx]
            y_train, y_val = y_pool[train_idx], y_pool[val_idx]
            
            X_train_processed, X_val_processed, X_test_processed = self.preprocess_data(
                X_train=X_train, X_val=X_val, X_test=X_test 
            )
                
            # Yield the fold to the experiment runner
            yield X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test
            
    def get_data_split(self, seed):
        """Create train/test/validation split with preprocessing"""
        # Split into 80% pool and 20% fixed test
        X_pool, X_test, y_pool, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=seed, stratify=self.y
        )
        
        # Split into 80% (of 80%) train and 20% (of 80%) validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_pool, y_pool, test_size=0.2, random_state=seed, stratify=y_pool
        )
        
        X_train_processed, X_val_processed, X_test_processed = self.preprocess_data(
            X_train=X_train, X_val=X_val, X_test=X_test
        )
        
        return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test
    

class SyntheticDataManager():
    """Generates synthetic data using scikit-learn's make_classification"""
    def __init__(self):
        # Dictionary of tuples (X, y)
        self.datasets_dict = {}
        
    def create_dataset(self, label, num_dims, n_samples=500, n_informative=4, 
                      n_classes=2, n_clusters_per_class=1, flip_y=0.01, 
                      class_sep=1.0, random_state=42):
        """
        Create and store a single synthetic dataset.
        
        Args:
            label: Unique identifier for this dataset
            num_dims: Number of features (dimensionality)
            n_samples: Total number of samples to generate
            n_informative: Number of informative features
            n_classes: Number of classes
            n_clusters_per_class: Number of clusters per class
            flip_y: Fraction of samples whose class is randomly flipped (noise)
            class_sep: Class separation factor (larger = easier classification)
            random_state: Random seed for reproducibility
        
        Returns:
            tuple: (X, y) arrays that were generated
        """
        # Validation
        if n_informative > num_dims:
            raise ValueError(
                f"n_informative ({n_informative}) cannot exceed num_dims ({num_dims}). "
                f"Set n_informative <= {num_dims}."
            )
        
        if label in self.datasets_dict:
            print(f"Warning: Dataset '{label}' already exists. Overwriting...")
        
        # Generate the synthetic dataset
        X, y = datasets.make_classification(
            n_samples=n_samples,
            n_features=num_dims,
            n_informative=n_informative,
            n_redundant=0,
            n_repeated=0,
            n_classes=n_classes,
            n_clusters_per_class=n_clusters_per_class,
            flip_y=flip_y,
            class_sep=class_sep,
            random_state=random_state
        )
        
        # Store in dictionary
        self.datasets_dict[label] = (X, y)
        
        return X, y
    
    def initialise_datasets(self, mode, num_dims, sweep_values, 
                           n_classes=2, n_samples=500, random_state=42):
        """
        Pre-generate multiple datasets for an experiment mode.
        
        Args:
            mode: Experiment mode ('feature_complexity', 'margin', 'clusters', 'noise')
            num_dims: Number of features
            sweep_values: List of parameter values to test
            n_classes: Number of classes (default: 2 for binary classification)
            n_samples: Number of samples per dataset
            random_state: Random seed
        
        Returns:
            dict: {label: (X, y)} mapping of generated datasets
        """
        # Map mode to parameter name
        mode_config = {
            'feature_complexity': 'n_informative',
            'margin': 'class_sep',
            'clusters': 'n_clusters_per_class',
            'noise': 'flip_y'
        }
        
        if mode not in mode_config:
            raise ValueError(
                f"Mode '{mode}' not supported. "
                f"Available modes: {list(mode_config.keys())}"
            )
        
        param_name = mode_config[mode]
                
        # Generate dataset for each experiment value
        for value in sweep_values:
            label = f"{mode}_{value}"
            
            # Build parameter dict with varying parameter
            params = {
                'label': label,
                'num_dims': num_dims,
                'n_samples': n_samples,
                'n_classes': n_classes,
                'random_state': random_state,
                param_name: value  # This is the varying parameter
            }
            
            self.create_dataset(**params)
        
        return self.datasets_dict
    
    def preprocess_data(self, X_train, X_val, X_test):
        preprocessing_pipeline = Pipeline([
            ('std_scaler', StandardScaler()), 
            ('minmax_scaler', MinMaxScaler(feature_range=(0, 1)))
        ])
        
        X_train_processed = preprocessing_pipeline.fit_transform(X_train)
        X_test_processed = preprocessing_pipeline.transform(X_test)
        X_val_processed = preprocessing_pipeline.transform(X_val)
        
        return X_train_processed, X_val_processed, X_test_processed
    
    def get_kfold_splits(self, seed, label, k_folds=5):
        """
        Generator that yields k mutually exclusive splits of the data.
        """
        X, y = self.datasets_dict[label]
        
        # Split into pool and test
        X_pool, X_test, y_pool, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        
        # Split X_pool into training and validation folds
        for train_idx, val_idx in skf.split(X_pool, y_pool):
            X_train, X_val = X_pool[train_idx], X_pool[val_idx]
            y_train, y_val = y_pool[train_idx], y_pool[val_idx]
            
            X_train_processed, X_val_processed, X_test_processed = self.preprocess_data(
                X_train=X_train, X_val=X_val, X_test=X_test
            )
                
            # Yield the fold to the experiment runner
            yield X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test
            
    def get_data_split(self, seed, label):
        """Create train/test/validation split with preprocessing"""
        # Split into 80% pool and 20% fixed test
        X, y = self.datasets_dict[label]
        
        X_pool, X_test, y_pool, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        # Split into 80% (of 80%) train and 20% (of 80%) validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_pool, y_pool, test_size=0.2, random_state=seed, stratify=y_pool
        )
        
        # Preprocessing
        X_train_processed, X_val_processed, X_test_processed = self.preprocess_data(
            X_train=X_train, X_val=X_val, X_test=X_test
        )
        
        return X_train_processed, X_val_processed, X_test_processed, y_train, y_val, y_test
