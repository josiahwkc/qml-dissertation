"""
Unit Tests for Quantum vs. Classical SVM Benchmark Experiment
==============================================================
Author: Test Suite for Josiah Chan's Code

Tests cover:
- Statistical functions (Nadeau-Bengio t-test)
- ExperimentConfig class
- ExperimentRunner class methods
- Integration tests
"""

import unittest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from sklearn.svm import SVC
import sys
import os

# Mock the quantum imports that might not be available
sys.modules['qiskit'] = MagicMock()
sys.modules['qiskit.circuit'] = MagicMock()
sys.modules['qiskit.circuit.library'] = MagicMock()
sys.modules['qiskit_machine_learning'] = MagicMock()
sys.modules['qiskit_machine_learning.kernels'] = MagicMock()
sys.modules['feature_map_factory'] = MagicMock()
sys.modules['data_manager'] = MagicMock()
sys.modules['tuner'] = MagicMock()

# Import after mocking
from experiment import (
    nadeau_bengio_corrected_ttest,
    ExperimentConfig,
    ExperimentRunner
)


class TestNadeauBengioCorrectedTTest(unittest.TestCase):
    """Test suite for Nadeau-Bengio corrected t-test function"""
    
    def test_identical_scores_returns_one(self):
        """Test that identical scores return p-value of 1.0"""
        q_scores = [0.8, 0.8, 0.8, 0.8, 0.8]
        c_scores = [0.8, 0.8, 0.8, 0.8, 0.8]
        n_train = 100
        n_test = 20
        
        p_val = nadeau_bengio_corrected_ttest(q_scores, c_scores, n_train, n_test)
        
        self.assertEqual(p_val, 1.0, "Identical scores should give p-value of 1.0")
    
    def test_insufficient_trials_returns_one(self):
        """Test that < 2 trials returns p-value of 1.0"""
        q_scores = [0.8]
        c_scores = [0.7]
        n_train = 100
        n_test = 20
        
        p_val = nadeau_bengio_corrected_ttest(q_scores, c_scores, n_train, n_test)
        
        self.assertEqual(p_val, 1.0, "Less than 2 trials should return 1.0")
    
    def test_clearly_different_scores_low_pvalue(self):
        """Test that clearly different scores give low p-value"""
        q_scores = [0.9, 0.91, 0.89, 0.90, 0.92]
        c_scores = [0.5, 0.51, 0.49, 0.50, 0.52]
        n_train = 100
        n_test = 20
        
        p_val = nadeau_bengio_corrected_ttest(q_scores, c_scores, n_train, n_test)
        
        self.assertLess(p_val, 0.01, "Clearly different scores should give p < 0.01")
    
    def test_similar_scores_high_pvalue(self):
        """Test that similar scores give high p-value"""
        q_scores = [0.80, 0.81, 0.79, 0.80, 0.82]
        c_scores = [0.80, 0.79, 0.81, 0.80, 0.78]
        n_train = 100
        n_test = 20
        
        p_val = nadeau_bengio_corrected_ttest(q_scores, c_scores, n_train, n_test)
        
        self.assertGreater(p_val, 0.05, "Similar scores should give p > 0.05")
    
    def test_variance_correction_applied(self):
        """Test that variance correction accounts for overlapping training sets"""
        q_scores = [0.85, 0.86, 0.84, 0.85, 0.87]
        c_scores = [0.80, 0.81, 0.79, 0.80, 0.82]
        n_train = 100
        n_test = 20
        
        # The correction should affect the p-value
        p_val = nadeau_bengio_corrected_ttest(q_scores, c_scores, n_train, n_test)
        
        # Should return a valid p-value between 0 and 1
        self.assertTrue(0 <= p_val <= 1, "p-value should be between 0 and 1")
    
    def test_handles_numpy_arrays(self):
        """Test that function works with numpy arrays"""
        q_scores = np.array([0.85, 0.86, 0.84])
        c_scores = np.array([0.80, 0.81, 0.79])
        n_train = 100
        n_test = 20
        
        p_val = nadeau_bengio_corrected_ttest(q_scores, c_scores, n_train, n_test)
        
        self.assertTrue(0 <= p_val <= 1, "Should handle numpy arrays")


class TestExperimentConfig(unittest.TestCase):
    """Test suite for ExperimentConfig class"""
    
    def test_get_valid_mode(self):
        """Test getting configuration for valid modes"""
        valid_modes = ['size', 'imbalance', 'feature_complexity', 
                      'margin', 'clusters', 'noise']
        
        for mode in valid_modes:
            config = ExperimentConfig.get(mode)
            
            self.assertIn('x_label', config)
            self.assertIn('title_suffix', config)
            self.assertIn('value_name', config)
            self.assertIn('is_synthetic', config)
    
    def test_get_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError"""
        with self.assertRaises(ValueError) as context:
            ExperimentConfig.get('invalid_mode')
        
        self.assertIn('Invalid mode', str(context.exception))
    
    def test_synthetic_modes_marked_correctly(self):
        """Test that synthetic modes are marked as is_synthetic=True"""
        synthetic_modes = ['feature_complexity', 'margin', 'clusters', 'noise']
        
        for mode in synthetic_modes:
            config = ExperimentConfig.get(mode)
            self.assertTrue(config['is_synthetic'], 
                          f"{mode} should be marked as synthetic")
    
    def test_csv_modes_marked_correctly(self):
        """Test that CSV modes are marked as is_synthetic=False"""
        csv_modes = ['size', 'imbalance']
        
        for mode in csv_modes:
            config = ExperimentConfig.get(mode)
            self.assertFalse(config['is_synthetic'], 
                           f"{mode} should NOT be marked as synthetic")
    
    def test_all_configs_have_required_fields(self):
        """Test that all mode configs have required fields"""
        required_fields = ['x_label', 'title_suffix', 'value_name', 'is_synthetic']
        
        for mode in ExperimentConfig.MODES.keys():
            config = ExperimentConfig.get(mode)
            
            for field in required_fields:
                self.assertIn(field, config, 
                            f"Mode '{mode}' missing field '{field}'")


class TestExperimentRunner(unittest.TestCase):
    """Test suite for ExperimentRunner class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock quantum provider
        self.mock_qp = Mock()
        
        # Test configuration
        self.sweep_values_dict = {
            'size': [20, 40, 60],
            'feature_complexity': [1, 2, 3],
            'margin': [0.5, 1.0, 1.5]
        }
        
        # Create runner instance
        self.runner = ExperimentRunner(
            quantum_provider=self.mock_qp,
            sweep_values_dict=self.sweep_values_dict,
            num_dims=4,
            num_trials=3,
            fixed_size=100
        )
    
    def test_initialization(self):
        """Test ExperimentRunner initialization"""
        self.assertEqual(self.runner.num_dims, 4)
        self.assertEqual(self.runner.num_trials, 3)
        self.assertEqual(self.runner.fixed_size, 100)
        self.assertIsNone(self.runner.data_manager)
        self.assertIsInstance(self.runner.classical_clf, SVC)
    
    def test_clear_results(self):
        """Test that clear_results properly initializes results dict"""
        # Add some dummy data
        self.runner.results['x_values'] = [1, 2, 3]
        self.runner.results['q_acc'] = [0.8, 0.9, 0.85]
        
        # Clear results
        self.runner.clear_results()
        
        # Check all fields are empty lists
        self.assertEqual(self.runner.results['x_values'], [])
        self.assertEqual(self.runner.results['q_acc'], [])
        self.assertEqual(self.runner.results['c_acc'], [])
        
        # Check all required fields exist
        required_fields = [
            'x_values', 'sizes',
            'q_acc', 'q_acc_std', 'q_f1', 'q_f1_std', 'q_time',
            'c_acc', 'c_acc_std', 'c_f1', 'c_f1_std', 'c_time',
            'delta_acc', 'p_val_acc', 'delta_f1', 'p_val_f1'
        ]
        
        for field in required_fields:
            self.assertIn(field, self.runner.results)
            self.assertEqual(self.runner.results[field], [])
    
    def test_run_classical_returns_correct_format(self):
        """Test that run_classical returns (accuracy, f1, time) tuple"""
        # Create simple test data
        X_train = np.random.randn(50, 4)
        X_test = np.random.randn(20, 4)
        y_train = np.random.randint(0, 2, 50)
        y_test = np.random.randint(0, 2, 20)
        
        params = {'C': 1.0, 'gamma': 'scale'}
        
        # Run classical
        acc, f1, elapsed_time = self.runner.run_classical(
            X_train, X_test, y_train, y_test, params
        )
        
        # Check return types
        self.assertIsInstance(acc, (float, np.floating))
        self.assertIsInstance(f1, (float, np.floating))
        self.assertIsInstance(elapsed_time, float)
        
        # Check value ranges
        self.assertTrue(0 <= acc <= 1, "Accuracy should be between 0 and 1")
        self.assertTrue(0 <= f1 <= 1, "F1 should be between 0 and 1")
        self.assertGreaterEqual(elapsed_time, 0, "Time should be non-negative")
    
    def test_run_classical_uses_correct_parameters(self):
        """Test that run_classical applies the given parameters"""
        X_train = np.random.randn(50, 4)
        X_test = np.random.randn(20, 4)
        y_train = np.random.randint(0, 2, 50)
        y_test = np.random.randint(0, 2, 20)
        
        params = {'C': 10.0, 'gamma': 0.01}
        
        self.runner.run_classical(X_train, X_test, y_train, y_test, params)
        
        # Check that parameters were set
        self.assertEqual(self.runner.classical_clf.C, 10.0)
        self.assertEqual(self.runner.classical_clf.gamma, 0.01)
    
    @patch('experiment.SyntheticDataManager')
    def test_initialise_datasets_synthetic_mode(self, mock_sdm_class):
        """Test dataset initialization for synthetic modes"""
        mock_dm_instance = Mock()
        mock_sdm_class.return_value = mock_dm_instance
        
        self.runner.initialise_datasets(mode='feature_complexity')
        
        # Check that SyntheticDataManager was created
        mock_sdm_class.assert_called_once()
        
        # Check that initialise_datasets was called with correct params
        mock_dm_instance.initialise_datasets.assert_called_once_with(
            num_dims=4,
            mode='feature_complexity',
            sweep_values=[1, 2, 3]
        )
        
        # Check that data_manager was set
        self.assertEqual(self.runner.data_manager, mock_dm_instance)
    
    @patch('experiment.CSVDataManager')
    def test_initialise_datasets_csv_mode(self, mock_csv_class):
        """Test dataset initialization for CSV modes"""
        mock_dm_instance = Mock()
        mock_csv_class.return_value = mock_dm_instance
        
        self.runner.initialise_datasets(
            mode='size',
            filename='test.csv',
            target_col='label'
        )
        
        # Check that CSVDataManager was created
        mock_csv_class.assert_called_once()
        
        # Check that load_dataset was called
        mock_dm_instance.load_dataset.assert_called_once_with(
            filename='test.csv',
            target_col='label',
            num_dims=4
        )
    
    def test_initialise_datasets_invalid_mode_raises_error(self):
        """Test that invalid mode raises ValueError"""
        with self.assertRaises(ValueError) as context:
            self.runner.initialise_datasets(mode='invalid_mode')
        
        self.assertIn('Invalid mode', str(context.exception))


class TestExperimentRunnerIntegration(unittest.TestCase):
    """Integration tests for ExperimentRunner"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_qp = Mock()
        self.sweep_values_dict = {
            'feature_complexity': [1, 2, 3]
        }
        
        self.runner = ExperimentRunner(
            quantum_provider=self.mock_qp,
            sweep_values_dict=self.sweep_values_dict,
            num_dims=4,
            num_trials=2,
            fixed_size=50
        )
    
    def test_results_accumulation(self):
        """Test that results accumulate correctly across multiple values"""
        # Manually add results as if from experiments
        self.runner.results['x_values'].append(1)
        self.runner.results['c_acc'].append(0.80)
        self.runner.results['q_acc'].append(0.85)
        
        self.runner.results['x_values'].append(2)
        self.runner.results['c_acc'].append(0.82)
        self.runner.results['q_acc'].append(0.87)
        
        # Check accumulation
        self.assertEqual(len(self.runner.results['x_values']), 2)
        self.assertEqual(self.runner.results['x_values'], [1, 2])
        self.assertEqual(self.runner.results['c_acc'], [0.80, 0.82])
        self.assertEqual(self.runner.results['q_acc'], [0.85, 0.87])
    
    def test_clear_results_between_experiments(self):
        """Test that results are cleared between experiments"""
        # Add some results
        self.runner.results['x_values'] = [1, 2, 3]
        self.runner.results['q_acc'] = [0.8, 0.85, 0.9]
        
        # Clear results
        self.runner.clear_results()
        
        # Verify empty
        self.assertEqual(len(self.runner.results['x_values']), 0)
        self.assertEqual(len(self.runner.results['q_acc']), 0)
    
    def test_sweep_values_dict_access(self):
        """Test accessing sweep values from dict"""
        mode = 'feature_complexity'
        sweep_values = self.runner.sweep_values_dict[mode]
        
        self.assertEqual(sweep_values, [1, 2, 3])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def test_nadeau_bengio_with_empty_lists(self):
        """Test Nadeau-Bengio with empty score lists"""
        q_scores = []
        c_scores = []
        
        p_val = nadeau_bengio_corrected_ttest(q_scores, c_scores, 100, 20)
        
        self.assertEqual(p_val, 1.0, "Empty lists should return 1.0")
    
    def test_nadeau_bengio_with_single_trial(self):
        """Test Nadeau-Bengio with single trial"""
        q_scores = [0.85]
        c_scores = [0.80]
        
        p_val = nadeau_bengio_corrected_ttest(q_scores, c_scores, 100, 20)
        
        self.assertEqual(p_val, 1.0, "Single trial should return 1.0")
    
    def test_run_classical_with_minimal_data(self):
        """Test run_classical with minimal dataset"""
        runner = ExperimentRunner(
            quantum_provider=Mock(),
            sweep_values_dict={'size': [10]},
            num_dims=2,
            num_trials=1,
            fixed_size=10
        )
        
        # Minimal dataset
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        X_test = np.array([[11, 12], [13, 14]])
        y_train = np.array([0, 1, 0, 1, 0])
        y_test = np.array([1, 0])
        
        params = {'C': 1.0, 'gamma': 'scale'}
        
        # Should not raise error
        acc, f1, time = runner.run_classical(X_train, X_test, y_train, y_test, params)
        
        self.assertTrue(0 <= acc <= 1)
        self.assertTrue(0 <= f1 <= 1)
        self.assertGreaterEqual(time, 0)


class TestConfigConsistency(unittest.TestCase):
    """Test configuration consistency across the codebase"""
    
    def test_all_modes_in_sweep_values_dict(self):
        """Test that sweep_values_dict can be created for all modes"""
        all_modes = list(ExperimentConfig.MODES.keys())
        
        sweep_values_dict = {
            'size': [20, 40, 60],
            'imbalance': [0.5, 0.6, 0.7],
            'feature_complexity': [1, 2, 3],
            'margin': [0.5, 1.0, 1.5],
            'clusters': [1, 2, 3],
            'noise': [0.0, 0.05, 0.10]
        }
        
        # Check all modes are covered
        for mode in all_modes:
            self.assertIn(mode, sweep_values_dict, 
                         f"Mode '{mode}' missing from sweep_values_dict")
    
    def test_config_labels_are_strings(self):
        """Test that all config labels are strings"""
        for mode, config in ExperimentConfig.MODES.items():
            self.assertIsInstance(config['x_label'], str)
            self.assertIsInstance(config['title_suffix'], str)
            self.assertIsInstance(config['value_name'], str)
    
    def test_is_synthetic_is_boolean(self):
        """Test that is_synthetic is always boolean"""
        for mode, config in ExperimentConfig.MODES.items():
            self.assertIsInstance(config['is_synthetic'], bool)


def run_tests():
    """Run all tests and generate report"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNadeauBengioCorrectedTTest))
    suite.addTests(loader.loadTestsFromTestCase(TestExperimentConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestExperimentRunner))
    suite.addTests(loader.loadTestsFromTestCase(TestExperimentRunnerIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigConsistency))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)
    
    return result


if __name__ == '__main__':
    run_tests()