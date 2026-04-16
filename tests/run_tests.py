"""
Test Runner for Quantum vs Classical SVM Benchmark
===================================================

Runs all pytest unit tests and integration tests, generates coverage report.

Usage:
    python run_tests.py
    python run_tests.py --verbose
    python run_tests.py --coverage
"""

import sys
import argparse
import pytest

def main():
    """Main test runner using pytest"""
    parser = argparse.ArgumentParser(description='Run tests for QSVM experiment')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--coverage', '-c', action='store_true',
                        help='Run with coverage report')
    parser.add_argument('--pattern', '-p', default='tests/',
                        help='Test directory or file pattern (default: tests/)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("QUANTUM VS CLASSICAL SVM BENCHMARK - TEST SUITE")
    print("="*80)
    print()

    # Build the pytest arguments dynamically
    pytest_args = [args.pattern]
    
    if args.verbose:
        pytest_args.append('-v')
        
    if args.coverage:
        # Assuming your production code is in 'src/'
        pytest_args.extend([
            '--cov=src', 
            '--cov-report=term-missing', 
            '--cov-report=html'
        ])

    # Run pytest
    exit_code = pytest.main(pytest_args)
    
    if args.coverage and exit_code == 0:
        print("\nHTML coverage report generated in the 'htmlcov/' directory.")
        print("Open htmlcov/index.html in your browser to view it.")
        
    sys.exit(exit_code)

if __name__ == '__main__':
    main()