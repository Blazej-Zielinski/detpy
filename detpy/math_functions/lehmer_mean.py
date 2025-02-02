import numpy as np

from detpy.math_functions.math_function import MathFunction


class LehmerMean(MathFunction):
    def __init__(self):
        super().__init__("Lehmer Mean")

    def evaluate(self, mut_factors: list[float], p: int = 2) -> float:
        """
        Calculate the Lehmer mean of a list of values.

        Parameters:
        - mut_factors (list[float]): The list of values.
        - p (int): The power parameter for the Lehmer mean.

        Returns: The Lehmer mean of the values in mut_factors.
        """
        numbers_array = np.array(mut_factors)
        numerator = np.sum(numbers_array ** p)
        denominator = np.sum(numbers_array ** (p - 1))
        return numerator / denominator
