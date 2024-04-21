import unittest
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

class TestIdealFunctionSelection(unittest.TestCase):
    def setUp(self):
        # Load test dataset
        self.test_dataset = pd.read_csv("test_data.csv")
        self.test_x = self.test_dataset.iloc[:, 0]
        self.test_y = self.test_dataset.iloc[:, 1]

        # Load training dataset with four functions
        self.training_dataset = pd.read_csv("train.csv")
        self.train_x = self.training_dataset.iloc[:, 0]
        self.train_y = self.training_dataset.iloc[:, 1:]

        # Load ideal functions
        self.ideal_functions = pd.read_csv("ideal.csv")

    def test_least_squares_selection(self):
        # Calculate sum of squared deviations for each ideal function
        errors = []
        for col in self.ideal_functions.columns[1:]:  # Exclude the first column (x-values)
            ideal_function = self.ideal_functions[col]
            sum_squared_deviations = 0
            for i in range(1, 5):
                # Interpolate ideal function values at training data x-values
                training_data = self.train_y[f"y{i}"]
                interpolated_values = np.interp(self.train_x, self.ideal_functions.iloc[:, 0], ideal_function)
                # Calculate sum of squared deviations
                sum_squared_deviations += np.sum((interpolated_values - training_data) ** 2)
            errors.append(sum_squared_deviations)

        # Select four ideal functions with the least sum of squared deviations
        best_fit_indices = np.argsort(errors)[:4]
        best_fit_functions = [self.ideal_functions.iloc[:, index + 1] for index in best_fit_indices]

        self.assertEqual(len(best_fit_functions), 4)
        self.assertIsInstance(best_fit_functions[0], pd.Series)
        self.assertEqual(len(best_fit_functions[0]), len(self.train_x))

    def test_test_errors(self):
        # Select ideal functions based on least squares criterion
        errors = []
        for col in self.ideal_functions.columns[1:]:
            ideal_function = self.ideal_functions[col]
            sum_squared_deviations = 0
            for i in range(1, 5):
                training_data = self.train_y[f"y{i}"]
                interpolated_values = np.interp(self.train_x, self.ideal_functions.iloc[:, 0], ideal_function)
                sum_squared_deviations += np.sum((interpolated_values - training_data) ** 2)
            errors.append(sum_squared_deviations)
        best_fit_indices = np.argsort(errors)[:4]
        best_fit_functions = [self.ideal_functions.iloc[:, index + 1] for index in best_fit_indices]

        # Test selected functions on test dataset and calculate test errors
        test_errors = []
        for best_fit_function in best_fit_functions:
            interpolated_values = np.interp(self.test_x, self.ideal_functions.iloc[:, 0], best_fit_function)
            test_error = mean_squared_error(self.test_y, interpolated_values)
            test_errors.append(test_error)

        # Ensure the test errors are calculated for all selected functions
        self.assertEqual(len(test_errors), 4)
        # Ensure test errors are non-negative
        self.assertTrue(all(error >= 0 for error in test_errors))


if __name__ == '__main__':
    unittest.main()
