"""
Integration Tests for Experiment Pipeline
==========================================

These tests verify the end-to-end workflow of the experiment system.

Aligned to the actual source API (experiment.py, data_manager.py, tuner.py).
Key facts about the real implementation:
  - ExperimentRunner.__init__ accepts only `quantum_provider`; num_dims /
    num_trials / fixed_size come from ExperimentConfig class constants.
  - run_quantum(X_train, X_test, y_train, y_test, kernel, params) — `kernel`
    is a positional argument, not a keyword.
  - results dict keys: x_values, q/c_acc/acc_std/f1/f1_std/time, delta_acc,
    p_val_acc, cohen_d_acc, delta_f1, p_val_f1, cohen_d_f1, time_ratio.
    There is no 'sizes' key.
  - _get_output_meta() takes no arguments (uses self.config internally).
  - initialise_datasets for sklearn modes creates a SyntheticDataManager and
    calls .initialise_datasets() on the *instance*, not the class.
"""

import unittest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
import sys
import types

from experiment import ExperimentRunner, ExperimentConfig


def _make_runner():
    """Return an ExperimentRunner with a minimal mock quantum provider."""
    mock_qp = Mock()
    mock_qp.backend = Mock()
    return ExperimentRunner(quantum_provider=mock_qp)


def _separable_data(n_per_class=25, n_features=4, seed=42):
    """Return a clearly linearly-separable binary dataset."""
    rng = np.random.default_rng(seed)
    offset = np.ones(n_features) * 3
    X = np.vstack([
        rng.standard_normal((n_per_class, n_features)) + offset,
        rng.standard_normal((n_per_class, n_features)) - offset,
    ])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


class TestFullExperimentPipeline(unittest.TestCase):
    """Test end-to-end initialisation paths for each data-source type."""

    def setUp(self):
        self.runner = _make_runner()


    @patch("experiment.SyntheticDataManager")
    def test_synthetic_experiment_initialises_data_manager(self, MockDM):
        """initialise_datasets for an sklearn mode creates a SyntheticDataManager
        and calls .initialise_datasets() on the instance with the right args."""
        mock_dm_instance = Mock()
        MockDM.return_value = mock_dm_instance

        self.runner.initialise_datasets(mode="feature_complexity")

        # ExperimentRunner should have stored the instance
        self.assertIsNotNone(self.runner.data_manager)
        self.assertIs(self.runner.data_manager, mock_dm_instance)

        # The instance method (not the class) should have been called once
        mock_dm_instance.initialise_datasets.assert_called_once()

    @patch("experiment.SyntheticDataManager")
    def test_synthetic_initialise_passes_correct_mode(self, MockDM):
        """The mode string is forwarded to SyntheticDataManager.initialise_datasets."""
        mock_dm_instance = Mock()
        MockDM.return_value = mock_dm_instance

        self.runner.initialise_datasets(mode="noise")

        _, kwargs = mock_dm_instance.initialise_datasets.call_args
        self.assertEqual(kwargs.get("mode") or
                         mock_dm_instance.initialise_datasets.call_args[0][0],
                         "noise")

    @patch("experiment.SyntheticDataManager")
    def test_synthetic_initialise_passes_num_dims(self, MockDM):
        """num_dims from ExperimentConfig.NUM_DIMS is forwarded correctly."""
        mock_dm_instance = Mock()
        MockDM.return_value = mock_dm_instance

        self.runner.initialise_datasets(mode="margin")

        call_kwargs = mock_dm_instance.initialise_datasets.call_args
        # Accept either positional or keyword
        all_args = list(call_kwargs[0]) + list(call_kwargs[1].values())
        self.assertIn(ExperimentConfig.NUM_DIMS, all_args)


    @patch("experiment.CSVDataManager")
    def test_csv_experiment_initialises_data_manager(self, MockDM):
        """initialise_datasets for a CSV mode creates a CSVDataManager and
        calls .load_dataset() with the supplied filename and target_col."""
        mock_dm_instance = Mock()
        MockDM.return_value = mock_dm_instance

        self.runner.initialise_datasets(
            mode="size",
            filename="test.csv",
            target_col="label",
        )

        self.assertIsNotNone(self.runner.data_manager)
        self.assertIs(self.runner.data_manager, mock_dm_instance)
        mock_dm_instance.load_dataset.assert_called_once_with(
            filename="test.csv",
            target_col="label",
            num_dims=ExperimentConfig.NUM_DIMS,
        )

    @patch("experiment.CSVDataManager")
    def test_csv_mode_without_filename_raises(self, _MockDM):
        """CSV mode must reject calls missing filename or target_col."""
        with self.assertRaises(ValueError):
            self.runner.initialise_datasets(mode="size")

    @patch("experiment.CSVDataManager")
    def test_csv_mode_without_target_col_raises(self, _MockDM):
        with self.assertRaises(ValueError):
            self.runner.initialise_datasets(mode="size", filename="data.csv")


    @patch("experiment.SyntheticDataManager")
    def test_config_stored_after_initialise(self, MockDM):
        """self.config must be populated after initialise_datasets."""
        MockDM.return_value = Mock()
        self.runner.initialise_datasets(mode="clusters")
        self.assertIsNotNone(self.runner.config)
        self.assertEqual(self.runner.config,
                         ExperimentConfig.get("clusters"))


class TestParameterValidation(unittest.TestCase):
    """Verify that ExperimentRunner picks up configuration from ExperimentConfig."""

    def test_num_dims_comes_from_experiment_config(self):
        runner = _make_runner()
        self.assertEqual(runner.num_dims, ExperimentConfig.NUM_DIMS)

    def test_num_trials_comes_from_experiment_config(self):
        runner = _make_runner()
        self.assertEqual(runner.num_trials, ExperimentConfig.NUM_TRIALS)

    def test_fixed_size_comes_from_experiment_config(self):
        runner = _make_runner()
        self.assertEqual(runner.fixed_size, ExperimentConfig.FIXED_SIZE)

    def test_runner_stores_quantum_provider(self):
        mock_qp = Mock()
        runner = ExperimentRunner(quantum_provider=mock_qp)
        self.assertIs(runner.qp, mock_qp)

    def test_data_manager_none_before_initialise(self):
        runner = _make_runner()
        self.assertIsNone(runner.data_manager)

    def test_invalid_mode_raises_value_error(self):
        runner = _make_runner()
        with self.assertRaises(ValueError):
            runner.initialise_datasets(mode="not_a_real_mode")


class TestResultsStorage(unittest.TestCase):
    """Test results dict structure, initial state, and accumulation."""

    # The exact keys present in ExperimentRunner.clear_results()
    REQUIRED_FIELDS = [
        "x_values",
        "q_acc", "q_acc_std", "q_f1", "q_f1_std", "q_time",
        "c_acc", "c_acc_std", "c_f1", "c_f1_std", "c_time",
        "delta_acc", "p_val_acc", "cohen_d_acc",
        "delta_f1",  "p_val_f1",  "cohen_d_f1",
        "time_ratio",
    ]

    def setUp(self):
        self.runner = _make_runner()

    def test_results_dict_has_all_required_fields(self):
        for field in self.REQUIRED_FIELDS:
            self.assertIn(field, self.runner.results,
                          f"Missing key in results dict: '{field}'")

    def test_all_results_fields_are_lists(self):
        for key, value in self.runner.results.items():
            self.assertIsInstance(value, list,
                                  f"results['{key}'] should be a list")

    def test_results_empty_on_init(self):
        for key, value in self.runner.results.items():
            self.assertEqual(len(value), 0,
                             f"results['{key}'] should be empty on init")

    def test_clear_results_resets_non_empty_dict(self):
        self.runner.results["x_values"].append(99)
        self.runner.results["c_acc"].append(0.5)
        self.runner.clear_results()
        for key, value in self.runner.results.items():
            self.assertEqual(len(value), 0,
                             f"clear_results() left data in '{key}'")

    def test_results_accumulate_across_values(self):
        for v in [1, 2, 3]:
            self.runner.results["x_values"].append(v)
            self.runner.results["c_acc"].append(0.80 + v * 0.01)
            self.runner.results["q_acc"].append(0.85 + v * 0.01)

        self.assertEqual(self.runner.results["x_values"], [1, 2, 3])
        self.assertAlmostEqual(self.runner.results["c_acc"][0], 0.81, places=5)
        self.assertAlmostEqual(self.runner.results["q_acc"][2], 0.88, places=5)

    def test_no_extra_undocumented_keys(self):
        """Catches accidental introduction of keys like 'sizes' that don't exist."""
        self.assertNotIn("sizes", self.runner.results,
                         "'sizes' is not a valid results key")


class TestClassicalSVMExecution(unittest.TestCase):
    """Test run_classical — the real method signature and return contract."""

    def setUp(self):
        self.runner = _make_runner()

    def test_run_classical_returns_three_values(self):
        X, y = _separable_data()
        result = self.runner.run_classical(X, X, y, y, {"C": 1.0, "gamma": "scale"})
        self.assertEqual(len(result), 3,
                         "run_classical must return (acc, f1, time)")

    def test_run_classical_accuracy_in_unit_interval(self):
        X, y = _separable_data()
        acc, f1, t = self.runner.run_classical(
            X, X, y, y, {"C": 1.0, "gamma": "scale"}
        )
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_run_classical_f1_in_unit_interval(self):
        X, y = _separable_data()
        acc, f1, t = self.runner.run_classical(
            X, X, y, y, {"C": 1.0, "gamma": "scale"}
        )
        self.assertGreaterEqual(f1, 0.0)
        self.assertLessEqual(f1, 1.0)

    def test_run_classical_time_positive(self):
        X, y = _separable_data()
        _, _, t = self.runner.run_classical(
            X, X, y, y, {"C": 1.0, "gamma": "scale"}
        )
        self.assertGreaterEqual(t, 0)

    def test_run_classical_applies_params_to_classifier(self):
        """set_params must be forwarded to the internal SVC instance."""
        X, y = _separable_data()
        self.runner.run_classical(X, X, y, y, {"C": 42.0, "gamma": 0.05})
        self.assertEqual(self.runner.classical_clf.C, 42.0)
        self.assertEqual(self.runner.classical_clf.gamma, 0.05)

    def test_run_classical_high_accuracy_on_separable_data(self):
        """With clearly separable data a tuned RBF SVM should achieve > 90%."""
        X, y = _separable_data(n_per_class=50)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(y))
        X, y = X[idx], y[idx]
        split = int(0.8 * len(y))
        acc, _, _ = self.runner.run_classical(
            X[:split], X[split:], y[:split], y[split:],
            {"C": 10.0, "gamma": "scale"},
        )
        self.assertGreater(acc, 0.90,
                           f"Expected >90% on separable data, got {acc:.2%}")


class TestMockQuantumKernel(unittest.TestCase):
    """Test run_quantum with the correct positional signature and mocked kernel."""

    def setUp(self):
        self.runner = _make_runner()

    def _make_kernel(self, n_train, n_test):
        """Mock kernel whose evaluate() returns identity for train, random for test."""
        mock_kernel = Mock()
        mock_kernel.evaluate.side_effect = [
            np.eye(n_train),                              # matrix_train
            np.random.default_rng(0).random((n_test, n_train)),  # matrix_test
        ]
        return mock_kernel

    def test_run_quantum_returns_three_values(self):
        X, y = _separable_data(n_per_class=25)
        split = 40
        kernel = self._make_kernel(split, len(y) - split)
        result = self.runner.run_quantum(
            X[:split], X[split:], y[:split], y[split:],
            kernel,                     # positional — not a keyword argument
            {"C": 1.0},
        )
        self.assertEqual(len(result), 3)

    def test_run_quantum_kernel_called_twice(self):
        """kernel.evaluate must be called exactly twice: once for train, once for test."""
        X, y = _separable_data(n_per_class=25)
        split = 40
        kernel = self._make_kernel(split, len(y) - split)
        with patch("builtins.print"):
            self.runner.run_quantum(
                X[:split], X[split:], y[:split], y[split:],
                kernel, {"C": 1.0},
            )
        self.assertEqual(kernel.evaluate.call_count, 2)

    def test_run_quantum_train_call_uses_x_vec_only(self):
        """First kernel call (train matrix) must not pass y_vec."""
        X, y = _separable_data(n_per_class=25)
        split = 40
        kernel = self._make_kernel(split, len(y) - split)
        with patch("builtins.print"):
            self.runner.run_quantum(
                X[:split], X[split:], y[:split], y[split:],
                kernel, {"C": 1.0},
            )
        first_call_kwargs = kernel.evaluate.call_args_list[0][1]
        self.assertNotIn("y_vec", first_call_kwargs,
                         "Train kernel call must not pass y_vec")

    def test_run_quantum_test_call_passes_y_vec(self):
        """Second kernel call (test matrix) must pass y_vec=X_train."""
        X, y = _separable_data(n_per_class=25)
        split = 40
        kernel = self._make_kernel(split, len(y) - split)
        with patch("builtins.print"):
            self.runner.run_quantum(
                X[:split], X[split:], y[:split], y[split:],
                kernel, {"C": 1.0},
            )
        second_call_kwargs = kernel.evaluate.call_args_list[1][1]
        self.assertIn("y_vec", second_call_kwargs,
                      "Test kernel call must include y_vec")

    def test_run_quantum_accuracy_in_unit_interval(self):
        X, y = _separable_data(n_per_class=25)
        split = 40
        kernel = self._make_kernel(split, len(y) - split)
        with patch("builtins.print"):
            acc, f1, t = self.runner.run_quantum(
                X[:split], X[split:], y[:split], y[split:],
                kernel, {"C": 1.0},
            )
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)

    def test_run_quantum_time_positive(self):
        X, y = _separable_data(n_per_class=25)
        split = 40
        kernel = self._make_kernel(split, len(y) - split)
        with patch("builtins.print"):
            _, _, t = self.runner.run_quantum(
                X[:split], X[split:], y[:split], y[split:],
                kernel, {"C": 1.0},
            )
        self.assertGreaterEqual(t, 0)


def run_integration_tests():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [
        TestFullExperimentPipeline,
        TestParameterValidation,
        TestResultsStorage,
        TestClassicalSVMExecution,
        TestMockQuantumKernel,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    run_integration_tests()