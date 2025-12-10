import numpy as np

class MathFunctions:
    @staticmethod
    def calculate_weighted_average(values, weights):
        """
        Calculate the weighted average of a set of values.

        Parameters:
        - values (np.ndarray): Array of values to average.
        - weights (np.ndarray): Array of weights corresponding to the values.

        Returns:
        - float: The weighted average.
        """
        return np.sum(weights * values) / np.sum(weights)

    @staticmethod
    def calculate_clipped_weighted_average(values, weights, min_val=0, max_val=1):
        """
        Calculate the weighted average of a set of values and clip the result.

        Parameters:
        - values (np.ndarray): Array of values to average.
        - weights (np.ndarray): Array of weights corresponding to the values.
        - min_val (float): Minimum value for clipping.
        - max_val (float): Maximum value for clipping.

        Returns:
        - float: The clipped weighted average.
        """
        weighted_avg = MathFunctions.calculate_weighted_average(values, weights)
        return np.clip(weighted_avg, min_val, max_val)

    @staticmethod
    def calculate_lehmer_mean(values, weights, p=2):
        """
        Calculate the Lehmer mean of a set of values.

        Parameters:
        - values (np.ndarray): Array of values.
        - weights (np.ndarray): Array of weights corresponding to the values.
        - p (float): The power parameter for the Lehmer mean.

        Returns:
        - float: The Lehmer mean.
        """
        numerator = np.sum(weights * values**p)
        denominator = np.sum(weights * values**(p - 1))
        return numerator / denominator if denominator != 0 else 0

    @staticmethod
    def calculate_special_weighted_average(values, weights):
        """
        Calculate a special weighted average where the numerator involves the square of the values.

        Parameters:
        - values (np.ndarray): Array of values to average.
        - weights (np.ndarray): Array of weights corresponding to the values.

        Returns:
        - float: The calculated special weighted average.
        """
        numerator = np.sum(weights * values * values)
        denominator = np.sum(weights * values)
        return numerator / denominator if denominator != 0 else 0