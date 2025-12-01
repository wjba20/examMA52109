###
## cluster_maker - test file for preprocessing module
## Will Avery - University of Bath
## Task 3) Practical Exam MA52109
### December 1st

import unittest
import numpy as np
import pandas as pd

from cluster_maker.preprocessing import select_features, standardise_features


class TestPreprocessing(unittest.TestCase):
    """Unit tests for preprocessing.py module."""

    def setUp(self):
        """Create test data for all tests."""
        # Create a realistic test DataFrame with mixed column types
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],  # Numeric but might be ID
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'category': ['A', 'B', 'A', 'B', 'C'],  # Non-numeric
            'text': ['foo', 'bar', 'baz', 'qux', 'quux']  # Non-numeric
        })

    def test_select_features_validates_numeric_columns(self):
        """
        Test that select_features correctly rejects non-numeric columns.
        
        REAL PROBLEM: In clustering, non-numeric features cause mathematical errors
        in distance calculations. This test ensures the function properly validates
        input types before proceeding with clustering algorithms.
        """
        # Attempt to select a non-numeric column
        with self.assertRaises(TypeError) as context:
            select_features(self.test_df, ['feature1', 'category'])
        
        # Verify the error message identifies the problematic column
        error_msg = str(context.exception)
        self.assertIn('category', error_msg)
        self.assertIn('not numeric', error_msg)

    def test_select_features_handles_missing_columns_appropriately(self):
        """
        Test that select_features correctly reports ALL missing columns, not just the first.
        
        REAL PROBLEM: Poor error messages that only report the first missing column
        lead to frustrating debugging cycles. This test ensures users get complete
        information about all missing columns at once.
        """
        # Test with multiple missing columns
        with self.assertRaises(KeyError) as context:
            select_features(self.test_df, ['feature1', 'nonexistent1', 'nonexistent2'])
        
        error_msg = str(context.exception)
        
        # Check that BOTH missing columns are reported
        self.assertIn('nonexistent1', error_msg)
        self.assertIn('nonexistent2', error_msg)
        
        # Verify existing column is not in error message
        self.assertNotIn('feature1', error_msg)

    def test_standardise_features_produces_zero_mean_unit_variance(self):
        """
        Test that standardise_features correctly transforms data with meaningful statistics.
        
        REAL PROBLEM: Incorrect standardization skews clustering results since
        k-means is sensitive to feature scales. This test verifies the transformation
        produces the expected statistical properties that clustering algorithms rely on.
        """
        # Create data with different means and variances to test scaling
        X = np.array([
            [1.0, 100.0],
            [2.0, 200.0],
            [3.0, 300.0],
            [4.0, 400.0],
            [5.0, 500.0]
        ])
        
        X_scaled = standardise_features(X)
        
        # Verify zero mean (allowing for floating point tolerance)
        column_means = X_scaled.mean(axis=0)
        np.testing.assert_array_almost_equal(column_means, [0.0, 0.0], decimal=10)
        
        # Verify unit variance (standard deviation = 1)
        column_stds = X_scaled.std(axis=0, ddof=0)  # Population standard deviation
        np.testing.assert_array_almost_equal(column_stds, [1.0, 1.0], decimal=10)


if __name__ == "__main__":
    unittest.main()