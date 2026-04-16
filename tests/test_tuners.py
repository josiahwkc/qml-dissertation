import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from tuner import ClassicalSVMTuner, QuantumSVMTuner
from helpers import make_binary_data, split


class TestClassicalSVMTuner:
    def setup_method(self):
        # Clear class-level cache between tests to prevent cross-contamination
        ClassicalSVMTuner._cached_params.clear()

    def _make_split(self, seed=0):
        X, y = make_binary_data(100, seed=seed)
        return split(X, y)

    def test_returns_dict_with_required_keys(self):
        X_tr, X_v, y_tr, y_v = self._make_split()
        params = ClassicalSVMTuner.get_best_params(X_tr, X_v, y_tr, y_v, verbose=False)
        assert "C" in params
        assert "gamma" in params

    def test_C_is_from_valid_grid(self):
        X_tr, X_v, y_tr, y_v = self._make_split()
        params = ClassicalSVMTuner.get_best_params(X_tr, X_v, y_tr, y_v, verbose=False)
        assert params["C"] in [0.1, 1, 10, 100]

    def test_gamma_is_from_valid_grid(self):
        X_tr, X_v, y_tr, y_v = self._make_split()
        params = ClassicalSVMTuner.get_best_params(X_tr, X_v, y_tr, y_v, verbose=False)
        assert params["gamma"] in ["scale", 0.001, 0.01, 0.1, 1]

    def test_caching_returns_same_object(self):
        X_tr, X_v, y_tr, y_v = self._make_split()
        p1 = ClassicalSVMTuner.get_best_params(
            X_tr, X_v, y_tr, y_v, cache_key="k1", verbose=False
        )
        p2 = ClassicalSVMTuner.get_best_params(
            X_tr, X_v, y_tr, y_v, cache_key="k1", verbose=False
        )
        assert p1 is p2, "Second call should return the exact cached object"

    def test_different_cache_keys_are_independent(self):
        X_tr, X_v, y_tr, y_v = self._make_split(seed=0)
        X_tr2, X_v2, y_tr2, y_v2 = self._make_split(seed=1)
        ClassicalSVMTuner.get_best_params(
            X_tr, X_v, y_tr, y_v, cache_key="A", verbose=False
        )
        ClassicalSVMTuner.get_best_params(
            X_tr2, X_v2, y_tr2, y_v2, cache_key="B", verbose=False
        )
        assert "A" in ClassicalSVMTuner._cached_params
        assert "B" in ClassicalSVMTuner._cached_params
        # Should be independent entries (not the same reference)
        assert (
            ClassicalSVMTuner._cached_params["A"]
            is not ClassicalSVMTuner._cached_params["B"]
        )

    def test_no_cache_key_does_not_populate_cache(self):
        X_tr, X_v, y_tr, y_v = self._make_split()
        ClassicalSVMTuner.get_best_params(X_tr, X_v, y_tr, y_v, verbose=False)
        assert len(ClassicalSVMTuner._cached_params) == 0

    def test_chosen_params_outperform_worst_case(self):
        """Best params should beat a deliberately bad configuration."""
        from sklearn.svm import SVC
        from sklearn import metrics

        X_tr, X_v, y_tr, y_v = self._make_split()
        best = ClassicalSVMTuner.get_best_params(X_tr, X_v, y_tr, y_v, verbose=False)

        # Score with tuned params
        clf_best = SVC(kernel="rbf", **best)
        clf_best.fit(X_tr, y_tr)
        acc_best = metrics.accuracy_score(y_v, clf_best.predict(X_v))

        # Score with an arbitrary poor configuration
        clf_bad = SVC(kernel="rbf", C=0.001, gamma=100)
        clf_bad.fit(X_tr, y_tr)
        acc_bad = metrics.accuracy_score(y_v, clf_bad.predict(X_v))

        assert acc_best >= acc_bad, (
            f"Tuned params ({acc_best:.3f}) should be >= poor params ({acc_bad:.3f})"
        )


class TestQuantumSVMTuner:

    def setup_method(self):
        QuantumSVMTuner._cached_params.clear()

    def _make_split(self, n=40, n_features=2, seed=0):
        X, y = make_binary_data(n, n_features=n_features, seed=seed)
        return split(X, y, val_frac=0.25)

    def _make_kernel_mock(self, train_size, val_size):
        """Returns a mock kernel that produces identity-like precomputed matrices."""
        mock_kernel = MagicMock()
        mock_kernel.evaluate.side_effect = lambda x_vec, y_vec=None: (
            np.eye(len(x_vec)) if y_vec is None
            else np.ones((len(x_vec), len(y_vec))) * 0.5
        )
        return mock_kernel

    def test_returns_required_keys(self):
        X_tr, X_v, y_tr, y_v = self._make_split()
        with patch(
            "tuner.FidelityStatevectorKernel",
            return_value=self._make_kernel_mock(len(X_tr), len(X_v)),
        ):
            params = QuantumSVMTuner.get_best_params(
                X_tr, X_v, y_tr, y_v, num_dims=2, verbose=False
            )
        assert {"reps", "entanglement", "C"}.issubset(params.keys())

    def test_reps_in_valid_grid(self):
        X_tr, X_v, y_tr, y_v = self._make_split()
        with patch(
            "tuner.FidelityStatevectorKernel",
            return_value=self._make_kernel_mock(len(X_tr), len(X_v)),
        ):
            params = QuantumSVMTuner.get_best_params(
                X_tr, X_v, y_tr, y_v, num_dims=2, verbose=False
            )
        assert params["reps"] in [1, 2, 3]

    def test_entanglement_in_valid_grid(self):
        X_tr, X_v, y_tr, y_v = self._make_split()
        with patch(
            "tuner.FidelityStatevectorKernel",
            return_value=self._make_kernel_mock(len(X_tr), len(X_v)),
        ):
            params = QuantumSVMTuner.get_best_params(
                X_tr, X_v, y_tr, y_v, num_dims=2, verbose=False
            )
        assert params["entanglement"] in ["linear", "full"]

    def test_C_in_valid_grid(self):
        X_tr, X_v, y_tr, y_v = self._make_split()
        with patch(
            "tuner.FidelityStatevectorKernel",
            return_value=self._make_kernel_mock(len(X_tr), len(X_v)),
        ):
            params = QuantumSVMTuner.get_best_params(
                X_tr, X_v, y_tr, y_v, num_dims=2, verbose=False
            )
        assert params["C"] in [0.1, 1, 10, 100]

    def test_caching_returns_same_object(self):
        X_tr, X_v, y_tr, y_v = self._make_split()
        mock_kernel = self._make_kernel_mock(len(X_tr), len(X_v))
        with patch("tuner.FidelityStatevectorKernel", return_value=mock_kernel):
            p1 = QuantumSVMTuner.get_best_params(
                X_tr, X_v, y_tr, y_v, num_dims=2, cache_key="qk1", verbose=False
            )
            p2 = QuantumSVMTuner.get_best_params(
                X_tr, X_v, y_tr, y_v, num_dims=2, cache_key="qk1", verbose=False
            )
        assert p1 is p2

    def test_different_cache_keys_are_independent(self):
        X_tr, X_v, y_tr, y_v = self._make_split()
        mock = self._make_kernel_mock(len(X_tr), len(X_v))
        with patch("tuner.FidelityStatevectorKernel", return_value=mock):
            QuantumSVMTuner.get_best_params(
                X_tr, X_v, y_tr, y_v, num_dims=2, cache_key="qa", verbose=False
            )
            QuantumSVMTuner.get_best_params(
                X_tr, X_v, y_tr, y_v, num_dims=2, cache_key="qb", verbose=False
            )
        assert "qa" in QuantumSVMTuner._cached_params
        assert "qb" in QuantumSVMTuner._cached_params
        assert (
            QuantumSVMTuner._cached_params["qa"]
            is not QuantumSVMTuner._cached_params["qb"]
        )

    def test_kernel_error_does_not_crash_tuner(self):
        """If kernel.evaluate raises ValueError, the tuner should skip and continue."""
        X_tr, X_v, y_tr, y_v = self._make_split()
        broken_kernel = MagicMock()
        broken_kernel.evaluate.side_effect = ValueError("Simulated kernel failure")

        # Tuner should complete without raising (it logs and skips failed configs)
        with patch("tuner.FidelityStatevectorKernel", return_value=broken_kernel):
            # Returns empty dict {} or partial result — should not raise
            try:
                QuantumSVMTuner.get_best_params(
                    X_tr, X_v, y_tr, y_v, num_dims=2, verbose=False
                )
            except Exception as exc:
                pytest.fail(
                    f"QuantumSVMTuner raised an exception on kernel failure: {exc}"
                )