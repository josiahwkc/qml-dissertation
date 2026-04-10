"""
Test Runner for Quantum vs Classical SVM Benchmark
===================================================

Runs all unit tests and integration tests, generates coverage report.

Usage:
    python run_tests.py
    python run_tests.py --verbose
    python run_tests.py --coverage
"""

import coverage
import sys
import unittest
import argparse
from io import StringIO


def discover_and_run_tests(verbosity=2, pattern='test_*.py'):
    """
    Discover and run all tests in the current directory.
    
    Args:
        verbosity: Test output verbosity (0, 1, or 2)
        pattern: File pattern for test discovery
    
    Returns:
        unittest.TestResult: Test results
    """
    # Discover all tests
    loader = unittest.TestLoader()
    start_dir = '.'
    suite = loader.discover(start_dir, pattern=pattern)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def print_summary(result):
    """Print a detailed test summary"""
    print("\n" + "="*80)
    print("TEST EXECUTION SUMMARY")
    print("="*80)
    
    # Overall stats
    print(f"\nTests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    # Success rate
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) 
                       / result.testsRun * 100)
        print(f"\nSuccess Rate: {success_rate:.2f}%")
    
    # Failures detail
    if result.failures:
        print("\n" + "-"*80)
        print("FAILURES:")
        print("-"*80)
        for test, traceback in result.failures:
            print(f"\n{test}:")
            print(traceback)
    
    # Errors detail
    if result.errors:
        print("\n" + "-"*80)
        print("ERRORS:")
        print("-"*80)
        for test, traceback in result.errors:
            print(f"\n{test}:")
            print(traceback)
    
    print("\n" + "="*80)
    
    # Return exit code
    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


def run_with_coverage():
    """Run tests with coverage report"""    
    # Create coverage object
    cov = coverage.Coverage()
    cov.start()
    
    # Run tests
    result = discover_and_run_tests(verbosity=2)
    
    # Stop coverage
    cov.stop()
    cov.save()
    
    # Print summary
    exit_code = print_summary(result)
    
    # Print coverage report
    print("\n" + "="*80)
    print("CODE COVERAGE REPORT")
    print("="*80)
    cov.report()
    
    # Generate HTML report
    print("\nGenerating HTML coverage report...")
    cov.html_report(directory='htmlcov')
    print("HTML report saved to: htmlcov/index.html")
    
    return exit_code


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description='Run tests for QSVM experiment')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--coverage', '-c', action='store_true',
                       help='Run with coverage report')
    parser.add_argument('--pattern', '-p', default='test_*.py',
                       help='Test file pattern (default: test_*.py)')
    
    args = parser.parse_args()
    
    # Determine verbosity
    verbosity = 2 if args.verbose else 1
    
    print("="*80)
    print("QUANTUM VS CLASSICAL SVM BENCHMARK - TEST SUITE")
    print("="*80)
    print()
    
    # Run tests
    if args.coverage:
        exit_code = run_with_coverage()
    else:
        result = discover_and_run_tests(verbosity=verbosity, pattern=args.pattern)
        exit_code = print_summary(result)
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()