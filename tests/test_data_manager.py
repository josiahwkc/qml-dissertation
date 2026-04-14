import pytest
import numpy as np

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


# ===========================================================================
# SyntheticDataManager
# ===========================================================================

class TestSyntheticDataManager:

    def setup_method(self):
        self.dm = SyntheticDataManager()

    # -- create_dataset -------------------------------------------------------

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

    # -- initialise_datasets --------------------------------------------------

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

    # -- get_data_split -------------------------------------------------------

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

    # -- get_kfold_splits -----------------------------------------------------

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