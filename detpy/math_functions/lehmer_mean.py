import numpy as np

from detpy.math_functions.math_function import MathFunction


class LehmerMean(MathFunction):
    def __init__(self):
        super().__init__("Lehmer Mean")

    def evaluate(self, values: list[float], weights: list[float], p: int = 2) -> float:
        """
        Calculate the weighted Lehmer mean.

        Parameters:
        - values (list[float]): The list of values.
        - weights (list[float]): The weights corresponding to the values.
        - p (int): The power parameter for the Lehmer mean.

        Returns:
        - float: The weighted Lehmer mean.
        """
        values_array = np.array(values)
        weights_array = np.array(weights)

        numerator = np.sum(weights_array * (values_array ** p))
        denominator = np.sum(weights_array * (values_array ** (p - 1)))

        return numerator / denominator if denominator != 0 else 0.0
