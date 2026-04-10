"""
Integration Tests for Experiment Pipeline
==========================================

These tests verify the end-to-end workflow of the experiment system.
"""

import unittest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys

# Mock quantum imports
sys.modules['qiskit'] = MagicMock()
sys.modules['qiskit.circuit'] = MagicMock()
sys.modules['qiskit.circuit.library'] = MagicMock()
sys.modules['qiskit_machine_learning'] = MagicMock()
sys.modules['qiskit_machine_learning.kernels'] = MagicMock()
sys.modules['feature_map_factory'] = MagicMock()
sys.modules['data_manager'] = MagicMock()
sys.modules['tuner'] = MagicMock()

from experiment import ExperimentRunner, ExperimentConfig


class TestFullExperimentPipeline(unittest.TestCase):
    """Test complete experiment pipeline from initialization to plotting"""
    
    def setUp(self):
        """Set up mock dependencies"""
        self.mock_qp = Mock()
        self.mock_qp.backend = Mock()
        
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
    
    @patch('experiment.ClassicalSVMTuner')
    @patch('experiment.QuantumSVMTuner')
    @patch('experiment.SyntheticDataManager')
    def test_synthetic_experiment_workflow(self, mock_dm_class, mock_q_tuner, mock_c_tuner):
        """Test full workflow for synthetic data experiment"""
        # Mock data manager
        mock_dm = Mock()
        mock_dm_class.return_value = mock_dm
        
        # Mock get_data_split to return valid data
        mock_dm.get_data_split.return_value = (
            np.random.randn(40, 4),  # X_train
            np.random.randn(10, 4),  # X_val
            np.random.randn(20, 4),  # X_test
            np.random.randint(0, 2, 40),  # y_train
            np.random.randint(0, 2, 10),   # y_val
            np.random.randint(0, 2, 20)    # y_test
        )
        
        # Mock tuners
        mock_c_tuner.get_best_params.return_value = {'C': 1.0, 'gamma': 'scale'}
        mock_q_tuner.get_best_params.return_value = {
            'reps': 2, 'entanglement': 'linear', 'C': 1.0
        }
        
        # Initialize datasets
        self.runner.initialise_datasets(mode='feature_complexity')
        
        # Verify data manager was initialized
        self.assertIsNotNone(self.runner.data_manager)
        mock_dm.initialise_datasets.assert_called_once()
    
    @patch('experiment.ClassicalSVMTuner')
    @patch('experiment.QuantumSVMTuner')
    @patch('experiment.CSVDataManager')
    def test_csv_experiment_workflow(self, mock_dm_class, mock_q_tuner, mock_c_tuner):
        """Test full workflow for CSV data experiment"""
        # Mock data manager
        mock_dm = Mock()
        mock_dm_class.return_value = mock_dm
        
        # Mock tuners
        mock_c_tuner.get_best_params.return_value = {'C': 1.0, 'gamma': 'scale'}
        mock_q_tuner.get_best_params.return_value = {
            'reps': 2, 'entanglement': 'linear', 'C': 1.0
        }
        
        # Initialize datasets
        runner = ExperimentRunner(
            quantum_provider=self.mock_qp,
            sweep_values_dict={'size': [20, 40, 60]},
            num_dims=4,
            num_trials=2,
            fixed_size=50
        )
        
        runner.initialise_datasets(
            mode='size',
            filename='test.csv',
            target_col='label'
        )
        
        # Verify CSV data manager was initialized
        self.assertIsNotNone(runner.data_manager)
        mock_dm.load_dataset.assert_called_once_with(
            filename='test.csv',
            target_col='label',
            num_dims=4
        )


class TestParameterValidation(unittest.TestCase):
    """Test parameter validation and error handling"""
    
    def test_num_dims_validation(self):
        """Test that num_dims is properly stored and used"""
        runner = ExperimentRunner(
            quantum_provider=Mock(),
            sweep_values_dict={'size': [20]},
            num_dims=5,
            num_trials=3,
            fixed_size=100
        )
        
        self.assertEqual(runner.num_dims, 5)
    
    def test_num_trials_validation(self):
        """Test that num_trials is properly stored"""
        runner = ExperimentRunner(
            quantum_provider=Mock(),
            sweep_values_dict={'size': [20]},
            num_dims=4,
            num_trials=10,
            fixed_size=100
        )
        
        self.assertEqual(runner.num_trials, 10)
    
    def test_fixed_size_validation(self):
        """Test that fixed_size is properly stored"""
        runner = ExperimentRunner(
            quantum_provider=Mock(),
            sweep_values_dict={'size': [20]},
            num_dims=4,
            num_trials=5,
            fixed_size=75
        )
        
        self.assertEqual(runner.fixed_size, 75)


class TestResultsStorage(unittest.TestCase):
    """Test results storage and accumulation"""
    
    def setUp(self):
        """Set up test runner"""
        self.runner = ExperimentRunner(
            quantum_provider=Mock(),
            sweep_values_dict={'feature_complexity': [1, 2, 3]},
            num_dims=4,
            num_trials=3,
            fixed_size=100
        )
    
    def test_results_dict_structure(self):
        """Test that results dict has all required fields"""
        required_fields = [
            'x_values', 'sizes',
            'q_acc', 'q_acc_std', 'q_f1', 'q_f1_std', 'q_time',
            'c_acc', 'c_acc_std', 'c_f1', 'c_f1_std', 'c_time',
            'delta_acc', 'p_val_acc', 'delta_f1', 'p_val_f1'
        ]
        
        for field in required_fields:
            self.assertIn(field, self.runner.results)
            self.assertIsInstance(self.runner.results[field], list)
    
    def test_results_cleared_on_init(self):
        """Test that results are empty on initialization"""
        for key, value in self.runner.results.items():
            self.assertEqual(len(value), 0, f"{key} should be empty on init")
    
    def test_results_persist_between_values(self):
        """Test that results accumulate across experiment values"""
        # Simulate storing results for multiple values
        for value in [1, 2, 3]:
            self.runner.results['x_values'].append(value)
            self.runner.results['c_acc'].append(0.8 + value * 0.01)
            self.runner.results['q_acc'].append(0.85 + value * 0.01)
        
        # Check accumulation
        self.assertEqual(len(self.runner.results['x_values']), 3)
        self.assertEqual(self.runner.results['x_values'], [1, 2, 3])
        
        # Check values are correct
        self.assertAlmostEqual(self.runner.results['c_acc'][0], 0.81, places=5)
        self.assertAlmostEqual(self.runner.results['q_acc'][2], 0.88, places=5)


class TestClassicalSVMExecution(unittest.TestCase):
    """Test classical SVM training and prediction"""
    
    def setUp(self):
        """Set up runner"""
        self.runner = ExperimentRunner(
            quantum_provider=Mock(),
            sweep_values_dict={'size': [20]},
            num_dims=4,
            num_trials=2,
            fixed_size=50
        )
    
    def test_classical_svm_trained_with_params(self):
        """Test that classical SVM is trained with given parameters"""
        X_train = np.random.randn(50, 4)
        X_test = np.random.randn(20, 4)
        y_train = np.random.randint(0, 2, 50)
        y_test = np.random.randint(0, 2, 20)
        
        params = {'C': 5.0, 'gamma': 0.1}
        
        acc, f1, time = self.runner.run_classical(
            X_train, X_test, y_train, y_test, params
        )
        
        # Verify parameters were set
        self.assertEqual(self.runner.classical_clf.C, 5.0)
        self.assertEqual(self.runner.classical_clf.gamma, 0.1)
    
    def test_classical_svm_predictions_valid(self):
        """Test that predictions are in valid range"""
        # Create separable data
        np.random.seed(42)
        X_train = np.vstack([
            np.random.randn(25, 4) + [2, 2, 2, 2],
            np.random.randn(25, 4) - [2, 2, 2, 2]
        ])
        y_train = np.array([0]*25 + [1]*25)
        
        X_test = np.vstack([
            np.random.randn(10, 4) + [2, 2, 2, 2],
            np.random.randn(10, 4) - [2, 2, 2, 2]
        ])
        y_test = np.array([0]*10 + [1]*10)
        
        params = {'C': 1.0, 'gamma': 'scale'}
        
        acc, f1, time = self.runner.run_classical(
            X_train, X_test, y_train, y_test, params
        )
        
        # Check metrics are valid
        self.assertTrue(0 <= acc <= 1)
        self.assertTrue(0 <= f1 <= 1)
        self.assertGreater(time, 0)


class TestMockQuantumKernel(unittest.TestCase):
    """Test quantum kernel evaluation with mocked quantum components"""
    
    def setUp(self):
        """Set up runner with mocked quantum components"""
        self.runner = ExperimentRunner(
            quantum_provider=Mock(),
            sweep_values_dict={'feature_complexity': [1]},
            num_dims=4,
            num_trials=2,
            fixed_size=50
        )
    
    def test_quantum_kernel_evaluation_called(self):
        """Test that quantum kernel evaluate is called correctly"""
        # Create mock kernel
        mock_kernel = Mock()
        mock_kernel.evaluate.side_effect = [
            np.eye(50),  # Training kernel (50x50)
            np.random.randn(20, 50)  # Test kernel (20x50)
        ]
        
        X_train = np.random.randn(50, 4)
        X_test = np.random.randn(20, 4)
        y_train = np.random.randint(0, 2, 50)
        y_test = np.random.randint(0, 2, 20)
        
        params = {'C': 1.0}
        
        # Run quantum
        with patch('builtins.print'):  # Suppress print statements
            acc, f1, time = self.runner.run_quantum(
                X_train, X_test, y_train, y_test,
                kernel=mock_kernel,
                params=params
            )
        
        # Verify kernel was called twice
        self.assertEqual(mock_kernel.evaluate.call_count, 2)
        
        # Check metrics
        self.assertTrue(0 <= acc <= 1)
        self.assertTrue(0 <= f1 <= 1)
        self.assertGreater(time, 0)


def run_integration_tests():
    """Run all integration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestFullExperimentPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestParameterValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestResultsStorage))
    suite.addTests(loader.loadTestsFromTestCase(TestClassicalSVMExecution))
    suite.addTests(loader.loadTestsFromTestCase(TestMockQuantumKernel))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    run_integration_tests()