
import pytest
import numpy as np
from unittest.mock import MagicMock

from experiment import ExperimentConfig, ExperimentRunner

class TestExperimentConfig:

    def test_get_returns_dict_for_valid_mode(self):
        cfg = ExperimentConfig.get("size")
        assert isinstance(cfg, dict)

    def test_get_raises_for_invalid_mode(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            ExperimentConfig.get("nonexistent_mode")

    def test_all_modes_have_required_keys(self):
        required = {"x_label", "title_suffix", "data_source", "sweep_values"}
        for mode_name, cfg in ExperimentConfig.MODES.items():
            missing = required - cfg.keys()
            assert not missing, f"Mode '{mode_name}' missing keys: {missing}"

    def test_csv_modes_require_file(self):
        for mode_name, cfg in ExperimentConfig.MODES.items():
            if cfg["data_source"] == ExperimentConfig.DATA_SOURCE_CSV:
                assert cfg.get("requires_file") is True, (
                    f"CSV mode '{mode_name}' should have requires_file=True"
                )

    def test_synthetic_modes_do_not_require_file(self):
        non_csv_sources = {
            ExperimentConfig.DATA_SOURCE_SKLEARN,
            ExperimentConfig.DATA_SOURCE_QISKIT,
        }
        for mode_name, cfg in ExperimentConfig.MODES.items():
            if cfg["data_source"] in non_csv_sources:
                assert cfg.get("requires_file") is False, (
                    f"Synthetic mode '{mode_name}' should have requires_file=False"
                )

    def test_sweep_values_are_non_empty(self):
        for mode_name, cfg in ExperimentConfig.MODES.items():
            sv = cfg["sweep_values"]
            if isinstance(sv, list):
                assert len(sv) > 0, f"Mode '{mode_name}' has empty sweep_values"


class TestNadeauBengioTtest:
    """Tests for ExperimentRunner._nadeau_bengio_corrected_ttest"""

    def setup_method(self):
        mock_qp = MagicMock()
        self.runner = ExperimentRunner(quantum_provider=mock_qp)

    def test_returns_float(self):
        q = [0.8, 0.82, 0.79, 0.81]
        c = [0.75, 0.76, 0.74, 0.77]
        result = self.runner._nadeau_bengio_corrected_ttest(q, c, n_train=80, n_test=20)
        assert isinstance(result, float)

    def test_p_value_in_unit_interval(self):
        q = [0.9, 0.88, 0.91, 0.89, 0.92]
        c = [0.6, 0.62, 0.59, 0.61, 0.63]
        p = self.runner._nadeau_bengio_corrected_ttest(q, c, n_train=80, n_test=20)
        assert 0.0 <= p <= 1.0

    def test_identical_scores_return_p_one(self):
        """When models are identical there is no difference to detect."""
        scores = [0.8, 0.8, 0.8, 0.8]
        p = self.runner._nadeau_bengio_corrected_ttest(
            scores, scores, n_train=80, n_test=20
        )
        assert p == 1.0

    def test_single_trial_returns_p_one(self):
        """Need at least 2 trials for variance estimation."""
        p = self.runner._nadeau_bengio_corrected_ttest(
            [0.9], [0.7], n_train=80, n_test=20
        )
        assert p == 1.0

    def test_large_difference_gives_small_p(self):
        """A clearly different pair of score lists should produce p < 0.05."""
        q = [0.95] * 10
        c = [0.50] * 10
        p = self.runner._nadeau_bengio_corrected_ttest(q, c, n_train=80, n_test=20)
        assert p < 0.05, f"Expected small p-value, got {p:.4f}"

    def test_p_value_is_two_tailed(self):
        """Swapping q and c should give the same p-value (two-tailed test)."""
        q = [0.85, 0.87, 0.86, 0.88]
        c = [0.70, 0.72, 0.71, 0.73]
        p_forward = self.runner._nadeau_bengio_corrected_ttest(
            q, c, n_train=80, n_test=20
        )
        p_reverse = self.runner._nadeau_bengio_corrected_ttest(
            c, q, n_train=80, n_test=20
        )
        assert abs(p_forward - p_reverse) < 1e-10


class TestComputeStats:
    """Tests for ExperimentRunner._compute_stats"""

    def setup_method(self):
        mock_qp = MagicMock()
        self.runner = ExperimentRunner(quantum_provider=mock_qp)

    def _make_data(self, n=5):
        rng = np.random.default_rng(0)
        return {
            "acc":  list(rng.uniform(0.7, 0.9, n)),
            "f1":   list(rng.uniform(0.7, 0.9, n)),
            "time": list(rng.uniform(0.1, 1.0, n)),
        }

    def test_returns_all_expected_keys(self):
        c_data = self._make_data()
        q_data = self._make_data()
        stats = self.runner._calculate_statistics(c_data, q_data)
        expected_keys = {
            "c_avg_acc", "c_std_acc", "c_avg_f1", "c_std_f1", "c_avg_time",
            "q_avg_acc", "q_std_acc", "q_avg_f1", "q_std_f1", "q_avg_time",
            "delta_acc", "delta_f1", "time_ratio",
            "cohen_d_acc", "cohen_d_f1",
            "p_val_acc", "p_val_f1",
        }
        missing = expected_keys - stats.keys()
        assert not missing, f"Missing keys in _compute_stats output: {missing}"

    def test_delta_acc_sign(self):
        """delta_acc = q_avg_acc - c_avg_acc"""
        c_data = {"acc": [0.6] * 5, "f1": [0.6] * 5, "time": [0.1] * 5}
        q_data = {"acc": [0.8] * 5, "f1": [0.8] * 5, "time": [1.0] * 5}
        stats = self.runner._calculate_statistics(c_data, q_data)
        assert stats["delta_acc"] == pytest.approx(0.2)

    def test_time_ratio_correct(self):
        c_data = {"acc": [0.7] * 5, "f1": [0.7] * 5, "time": [1.0] * 5}
        q_data = {"acc": [0.7] * 5, "f1": [0.7] * 5, "time": [10.0] * 5}
        stats = self.runner._calculate_statistics(c_data, q_data)
        assert stats["time_ratio"] == pytest.approx(10.0)

    def test_zero_classical_time_avoids_division_error(self):
        c_data = {"acc": [0.7] * 5, "f1": [0.7] * 5, "time": [0.0] * 5}
        q_data = {"acc": [0.7] * 5, "f1": [0.7] * 5, "time": [5.0] * 5}
        stats = self.runner._calculate_statistics(c_data, q_data)
        assert stats["time_ratio"] == pytest.approx(5.0)

    def test_averages_are_correct(self):
        accs = [0.70, 0.75, 0.80]
        c_data = {"acc": accs, "f1": accs, "time": [1.0] * 3}
        q_data = {"acc": accs, "f1": accs, "time": [1.0] * 3}
        stats = self.runner._calculate_statistics(c_data, q_data)
        assert stats["c_avg_acc"] == pytest.approx(np.mean(accs))
        assert stats["q_avg_acc"] == pytest.approx(np.mean(accs))

    def test_std_is_non_negative(self):
        c_data = self._make_data()
        q_data = self._make_data()
        stats = self.runner._calculate_statistics(c_data, q_data)
        assert stats["c_std_acc"] >= 0
        assert stats["q_std_acc"] >= 0


# ===========================================================================
# ExperimentRunner.initialise_datasets (validation logic only)
# ===========================================================================

class TestExperimentRunnerInit:

    def setup_method(self):
        mock_qp = MagicMock()
        self.runner = ExperimentRunner(quantum_provider=mock_qp)

    def test_csv_mode_raises_without_filename(self):
        with pytest.raises(ValueError, match="requires filename"):
            self.runner.initialise_datasets(mode="size")

    def test_csv_mode_raises_without_target_col(self):
        with pytest.raises(ValueError, match="requires filename"):
            self.runner.initialise_datasets(mode="size", filename="data.csv")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid mode"):
            self.runner.initialise_datasets(mode="bad_mode")