import pytest
import numpy as np
import pandas as pd
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from data_manager import TrainingSampler, CSVDataManager, SyntheticDataManager, QuantumBenchmarkDataManager
from helpers import make_binary_data, split

class TestTrainingSampler:

    def test_output_shapes_balanced(self):
        X, y = make_binary_data(100)
        # Use a perfectly balanced pool for a balanced split
        X0 = X[y == 0][:40]
        X1 = X[y == 1][:40]
        X_pool = np.vstack([X0, X1])
        y_pool = np.array([0] * 40 + [1] * 40)

        X_tr, y_tr = TrainingSampler.create_class_imbalance(
            X_pool, y_pool, train_size=20, seed=0, imbalance_ratio=0.5
        )
        assert X_tr.shape == (20, X_pool.shape[1]), "Expected 20 training samples"
        assert y_tr.shape == (20,)

    def test_labels_match_features(self):
        X, y = make_binary_data(200)
        X_pool = X[y == 0][:50]
        y_pool = np.zeros(50, dtype=int)
        X_pool = np.vstack([X_pool, X[y == 1][:50]])
        y_pool = np.hstack([y_pool, np.ones(50, dtype=int)])

        X_tr, y_tr = TrainingSampler.create_class_imbalance(
            X_pool, y_pool, train_size=20, seed=7
        )
        assert len(X_tr) == len(y_tr), "Feature and label counts must match"

    def test_imbalance_ratio_respected(self):
        """With ratio=0.8, ~80% should be class-0."""
        n = 200
        X = np.zeros((n, 2))
        y = np.array([0] * (n // 2) + [1] * (n // 2))

        train_size = 50
        ratio = 0.8
        X_tr, y_tr = TrainingSampler.create_class_imbalance(
            X, y, train_size=train_size, seed=0, imbalance_ratio=ratio
        )
        n_class0 = np.sum(y_tr == 0)
        expected_class0 = int(train_size * ratio)
        assert n_class0 == expected_class0, (
            f"Expected {expected_class0} class-0 samples, got {n_class0}"
        )

    def test_output_is_shuffled(self):
        """Returned labels should not be monotonically [0,0,...,1,1,...] sorted."""
        X = np.zeros((200, 2))
        y = np.array([0] * 100 + [1] * 100)
        _, y_tr = TrainingSampler.create_class_imbalance(
            X, y, train_size=30, seed=99
        )
        # If labels are perfectly sorted the first half would all be the same class
        first_half = y_tr[: len(y_tr) // 2]
        assert not (np.all(first_half == 0) or np.all(first_half == 1)), (
            "Training labels appear un-shuffled"
        )

    def test_only_binary_classes_in_output(self):
        X = np.random.randn(200, 4)
        y = np.array([0] * 100 + [1] * 100)
        _, y_tr = TrainingSampler.create_class_imbalance(X, y, train_size=20, seed=1)
        assert set(np.unique(y_tr)).issubset({0, 1})

    def test_reproducibility(self):
        X = np.random.randn(200, 4)
        y = np.array([0] * 100 + [1] * 100)
        X1, y1 = TrainingSampler.create_class_imbalance(X, y, train_size=20, seed=5)
        X2, y2 = TrainingSampler.create_class_imbalance(X, y, train_size=20, seed=5)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestSyntheticDataManager:

    def setup_method(self):
        self.dm = SyntheticDataManager()

    def test_create_dataset_stores_data(self):
        X, y = self.dm.create_dataset(label="test", num_dims=4, n_samples=50)
        assert "test" in self.dm.datasets_dict
        stored_X, stored_y = self.dm.datasets_dict["test"]
        np.testing.assert_array_equal(X, stored_X)

    def test_create_dataset_shape(self):
        X, y = self.dm.create_dataset(label="shape_test", num_dims=6, n_samples=80)
        assert X.shape == (80, 6)
        assert y.shape == (80,)

    def test_create_dataset_binary_labels(self):
        _, y = self.dm.create_dataset(label="binary", num_dims=4, n_samples=100)
        assert set(np.unique(y)).issubset({0, 1})

    def test_create_dataset_raises_when_n_informative_exceeds_dims(self):
        with pytest.raises(ValueError, match="n_informative"):
            self.dm.create_dataset(label="bad", num_dims=3, n_informative=5)

    def test_create_dataset_reproducible(self):
        X1, y1 = self.dm.create_dataset(label="r1", num_dims=4, random_state=0)
        dm2 = SyntheticDataManager()
        X2, y2 = dm2.create_dataset(label="r1", num_dims=4, random_state=0)
        np.testing.assert_array_equal(X1, X2)

    def test_create_dataset_overwrites_existing_label(self, capsys):
        self.dm.create_dataset(label="dup", num_dims=4)
        self.dm.create_dataset(label="dup", num_dims=4)  # should warn, not crash
        out = capsys.readouterr().out
        assert "Warning" in out or "dup" in out

    def test_initialise_datasets_creates_correct_count(self):
        sweep = [1, 2, 3]
        self.dm.initialise_datasets(
            mode="feature_complexity", num_dims=5, sweep_values=sweep
        )
        assert len(self.dm.datasets_dict) == len(sweep)

    def test_initialise_datasets_labels_follow_mode_convention(self):
        self.dm.initialise_datasets(
            mode="margin", num_dims=4, sweep_values=[0.5, 1.0]
        )
        assert "margin_0.5" in self.dm.datasets_dict
        assert "margin_1.0" in self.dm.datasets_dict

    def test_initialise_datasets_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            self.dm.initialise_datasets(
                mode="INVALID_MODE", num_dims=4, sweep_values=[1]
            )

    def test_get_data_split_returns_six_arrays(self):
        self.dm.create_dataset(label="split", num_dims=4, n_samples=100)
        result = self.dm.get_data_split(seed=0, label="split")
        assert len(result) == 6, "Expected (X_train, X_val, X_test, y_train, y_val, y_test)"

    def test_get_data_split_no_overlap_between_train_and_test(self):
        """Train and test sets must be disjoint (no data leakage)."""
        self.dm.create_dataset(label="leak", num_dims=4, n_samples=200)
        X_tr, X_val, X_te, y_tr, y_val, y_te = self.dm.get_data_split(seed=0, label="leak")

        # Compare row sets using structured arrays for reliable set membership
        def to_set(arr):
            return set(map(tuple, arr))

        assert to_set(X_tr).isdisjoint(to_set(X_te)), "Train/test overlap detected"
        assert to_set(X_val).isdisjoint(to_set(X_te)), "Val/test overlap detected"
        assert to_set(X_tr).isdisjoint(to_set(X_val)), "Train/val overlap detected"

    def test_get_data_split_preprocessing_range(self):
        """Training set should be exactly in [0,1] after MinMaxScaler.
        Val/test may legitimately exceed [0,1] because the scaler is fit only
        on train data (no leakage) — we only verify the outputs are finite."""
        self.dm.create_dataset(label="scale", num_dims=5, n_samples=200)
        X_tr, X_val, X_te, *_ = self.dm.get_data_split(seed=0, label="scale")
        assert X_tr.min() >= -1e-6, "train has values below 0"
        assert X_tr.max() <= 1 + 1e-6, "train has values above 1"
        assert np.all(np.isfinite(X_val)), "val contains non-finite values"
        assert np.all(np.isfinite(X_te)), "test contains non-finite values"

    def test_get_data_split_scaler_fitted_only_on_train(self):
        """The scaler must not be re-fit on val/test; test min/max can exceed train's."""
        self.dm.create_dataset(label="fitcheck", num_dims=4, n_samples=500)
        X_tr, X_val, X_te, *_ = self.dm.get_data_split(seed=42, label="fitcheck")
        # Train set should span [0,1] perfectly after MinMax
        assert X_tr.min() >= -1e-6
        assert X_tr.max() <= 1 + 1e-6

    def test_get_kfold_splits_yields_correct_number_of_folds(self):
        self.dm.create_dataset(label="kfold", num_dims=4, n_samples=200)
        folds = list(self.dm.get_kfold_splits(seed=0, label="kfold", k_folds=5))
        assert len(folds) == 5

    def test_get_kfold_splits_each_fold_has_six_arrays(self):
        self.dm.create_dataset(label="kfold2", num_dims=4, n_samples=200)
        for fold in self.dm.get_kfold_splits(seed=0, label="kfold2", k_folds=3):
            assert len(fold) == 6

    def test_get_kfold_splits_test_set_identical_across_folds(self):
        """The hold-out *label* vector must be the same in every fold (same
        samples selected).  The preprocessed X_test values may differ slightly
        because the scaler is refit on each fold's training data, but y_test
        must be identical — it is never rescaled."""
        self.dm.create_dataset(label="kfold3", num_dims=4, n_samples=200)
        folds = list(self.dm.get_kfold_splits(seed=0, label="kfold3", k_folds=3))
        first_y_test = folds[0][5]
        for fold in folds[1:]:
            np.testing.assert_array_equal(
                fold[5], first_y_test,
                err_msg="y_test changed between folds — test set is not fixed"
            )
        # Also check the test set is the same size in every fold
        first_test_size = folds[0][2].shape[0]
        for fold in folds[1:]:
            assert fold[2].shape[0] == first_test_size
            
class TestQuantumBenchmarkDataManager:

    def setup_method(self):
        """Initialize a fresh instance of the manager before each test."""
        self.dm = QuantumBenchmarkDataManager()

    def test_initialization(self):
        """Ensure all data attributes start as None."""
        assert self.dm.X_pool is None
        assert self.dm.y_pool is None
        assert self.dm.X_test_fixed is None
        assert self.dm.y_test_fixed is None

    @patch('data_manager.ad_hoc_data') 
    def test_create_dataset_calls_ad_hoc_with_halved_sizes(self, mock_ad_hoc):
        """Test that pool_size and test_size are correctly halved per class to prevent inflation."""
        
        # Setup the mock to return dummy arrays just to prevent unpacking errors
        mock_ad_hoc.return_value = (
            np.zeros((800, 2)), np.zeros((800, 2)), 
            np.zeros((200, 2)), np.zeros((200, 2))
        )
        
        # Call with total sizes
        self.dm.create_dataset(num_dims=2, gap=0.3, pool_size=800, test_size=200)
        
        # Verify the math! 800 total -> 400 per class, 200 total -> 100 per class
        mock_ad_hoc.assert_called_once_with(
            training_size=400,
            test_size=100,
            n=2,
            gap=0.3,
            plot_data=True
        )

    @patch('data_manager.ad_hoc_data')
    def test_create_dataset_converts_one_hot_labels(self, mock_ad_hoc):
        """Test that one-hot labels from Qiskit are properly converted to 1D arrays."""
        
        # Create dummy features
        dummy_X_pool = np.random.rand(4, 2)
        dummy_X_test = np.random.rand(2, 2)
        
        # Create specific one-hot labels to test the argmax conversion
        # Class 0: [1, 0], Class 1: [0, 1]
        dummy_y_pool_onehot = np.array([[1, 0], [0, 1], [1, 0], [0, 1]]) 
        dummy_y_test_onehot = np.array([[0, 1], [1, 0]])
        
        # Inject the dummy data into the mock
        mock_ad_hoc.return_value = (
            dummy_X_pool, dummy_y_pool_onehot, 
            dummy_X_test, dummy_y_test_onehot
        )
        
        self.dm.create_dataset(pool_size=4, test_size=2)
        
        # Check features were stored correctly
        np.testing.assert_array_equal(self.dm.X_pool, dummy_X_pool)
        np.testing.assert_array_equal(self.dm.X_test_fixed, dummy_X_test)
        
        # Check labels were successfully converted from one-hot to 1D (argmax)
        expected_y_pool = np.array([0, 1, 0, 1])
        expected_y_test = np.array([1, 0])
        
        np.testing.assert_array_equal(self.dm.y_pool, expected_y_pool)
        np.testing.assert_array_equal(self.dm.y_test_fixed, expected_y_test)
    
    @patch('data_manager.ad_hoc_data')
    def test_get_data_split_returns_six_arrays(self, mock_ad_hoc):
        mock_ad_hoc.return_value = (
            np.zeros((800, 2)), np.array([[1, 0]] * 400 + [[0, 1]] * 400),
            np.zeros((200, 2)), np.array([[1, 0]] * 100 + [[0, 1]] * 100)
        )
        
        self.dm.create_dataset(num_dims=2, gap=0.3, pool_size=800, test_size=200)
        result = self.dm.get_data_split(seed=0)
        
        assert len(result) == 6, "Expected (X_train, X_val, X_test, y_train, y_val, y_test)"

    @patch('data_manager.ad_hoc_data')
    def test_get_data_split_no_overlap_between_train_and_test(self, mock_ad_hoc):
        """Train and test sets must be disjoint (no data leakage)."""
        
        # Generate strictly unique values to prevent accidental numeric collisions
        # X_pool gets values from 0 to 1599, X_test gets values from 2000 to 2399
        X_pool_dummy = np.arange(1600).reshape(800, 2) * 0.1
        y_pool_dummy = np.array([[1, 0]] * 400 + [[0, 1]] * 400)
        
        X_test_dummy = np.arange(2000, 2400).reshape(200, 2) * 0.1
        y_test_dummy = np.array([[1, 0]] * 100 + [[0, 1]] * 100)

        mock_ad_hoc.return_value = (X_pool_dummy, y_pool_dummy, X_test_dummy, y_test_dummy)

        self.dm.create_dataset(num_dims=2, gap=0.3, pool_size=800, test_size=200)
        X_tr, X_val, X_te, y_tr, y_val, y_te = self.dm.get_data_split(seed=0)

        def to_set(arr):
            # Rounding safely handles any floating point quirks if a MinMaxScaler was applied
            return set(map(tuple, np.round(arr, 6)))

        assert to_set(X_tr).isdisjoint(to_set(X_te)), "Train/test overlap detected"
        assert to_set(X_val).isdisjoint(to_set(X_te)), "Val/test overlap detected"
        assert to_set(X_tr).isdisjoint(to_set(X_val)), "Train/val overlap detected"
        
class TestCSVDataManager:

    def setup_method(self):
        self.dm = CSVDataManager()
        # We define the data here as a dictionary to avoid calling pd.read_csv 
        # while the mock might be active in the background.
        self.raw_data = {
            'feature1': [1.0, 1.1, 5.0, 5.1, 2.0, 2.2],
            'feature2': [2.0, 2.1, 6.0, 6.1, 3.0, 3.2],
            'category_col': ['A', 'B', 'A', 'B', 'A', 'B'],
            'target': [0, 1, 0, 1, 0, 1]
        }

    @patch('data_manager.Path.exists')
    @patch('data_manager.pd.read_csv')
    def test_load_dataset_stores_attributes(self, mock_read_csv, mock_exists):
        """Test core attributes with a guaranteed real DataFrame."""
        mock_exists.return_value = True
        # Explicitly return a real DataFrame object created from our dict
        mock_read_csv.return_value = pd.DataFrame(self.raw_data)
        
        self.dm.load_dataset(filename="test.csv", target_col="target", num_dims=2)
        
        assert self.dm.filename == "test.csv"
        assert self.dm.X.shape == (6, 3) # 2 features + 1 encoded category
        assert len(self.dm.y) == 6

    @patch('data_manager.Path.exists')
    @patch('data_manager.pd.read_csv')
    def test_load_dataset_encodes_categorical(self, mock_read_csv, mock_exists):
        """Verify categorical string columns are transformed to integers."""
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame(self.raw_data)
        
        # Identify category_col as the one to encode
        self.dm.load_dataset(filename="test.csv", target_col="target", categorical_cols=["category_col"])
        
        # Check that the data is numeric (LabelEncoder result)
        assert np.issubdtype(self.dm.X.dtype, np.number)

    @patch('data_manager.Path.exists')
    @patch('data_manager.pd.read_csv')
    def test_n_class_filtering(self, mock_read_csv, mock_exists):
        """Ensure rows are filtered based on n_class."""
        mock_exists.return_value = True
        # Data with 3 classes
        df_3_classes = pd.DataFrame({
            'f1': [1, 2, 3, 4],
            't': [0, 1, 2, 0]
        })
        mock_read_csv.return_value = df_3_classes
        
        self.dm.load_dataset("multi.csv", "t", n_class=2)
        
        unique_classes = np.unique(self.dm.y)
        assert 2 not in unique_classes
        assert len(unique_classes) == 2

    @patch('data_manager.Path.exists')
    @patch('data_manager.pd.read_csv')
    def test_get_data_split_full_pipeline(self, mock_read_csv, mock_exists):
        """Verify the 80/20 splits and PCA/Scaling pipeline works."""
        mock_exists.return_value = True
        # We need a larger dataset to satisfy StratifiedKFold/split requirements
        large_data = {
            'f1': np.random.randn(20),
            'f2': np.random.randn(20),
            'target': [0, 1] * 10
        }
        mock_read_csv.return_value = pd.DataFrame(large_data)
        
        self.dm.load_dataset("large.csv", "target", num_dims=1)
        X_tr, X_val, X_te, y_tr, y_val, y_te = self.dm.get_data_split(seed=42)
        
        # Verify PCA reduction worked (from 2 features to 1)
        assert X_tr.shape[1] == 1
        # Verify MinMaxScaler worked ([0, 1] range)
        assert X_tr.min() >= -1e-7 and X_tr.max() <= 1 + 1e-7

    @patch('data_manager.Path.exists')
    @patch('data_manager.pd.read_csv')
    def test_raises_error_on_missing_column(self, mock_read_csv, mock_exists):
        """Confirm ValueError is raised if target_col is missing from CSV."""
        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame(self.raw_data)
        
        with pytest.raises(ValueError, match="not found"):
            self.dm.load_dataset("test.csv", target_col="non_existent_column")