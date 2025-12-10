import numpy as np


class RandomValueGenerator:
    """
    A general class to generate random values based on specific criteria.
    """

    def generate_count_from_percentage(self, total_size: int, min_percentage: float, max_percentage: float) -> int:
        """
        Generate a random count based on a percentage range.

        Parameters:
        - total_size (int): The total size to calculate the count from.
        - min_percentage (float): The minimum percentage (e.g., 0.01 for 1%).
        - max_percentage (float): The maximum percentage (e.g., 0.2 for 20%).

        Returns:
        - int: The calculated random count.
        """
        if total_size <= 0:
            raise ValueError("Total size must be greater than 0.")
        if not (0 <= min_percentage <= max_percentage <= 1):
            raise ValueError("Percentages must be between 0 and 1, and min_percentage <= max_percentage.")

        random_percentage = np.random.uniform(min_percentage, max_percentage)
        return int(random_percentage * total_size)

    def generate_normal(self, mean: float, std_dev: float, min_val: float = None, max_val: float = None) -> float:
        """
        Generate a random value from a normal distribution.

        Parameters:
        - mean (float): The mean of the normal distribution.
        - std_dev (float): The standard deviation of the normal distribution.
        - min_val (float, optional): Minimum value to clip the result.
        - max_val (float, optional): Maximum value to clip the result.

        Returns:
        - float: A random value from the normal distribution.
        """
        value = np.random.normal(mean, std_dev)
        if min_val is not None or max_val is not None:
            value = np.clip(value, min_val, max_val)
        return value

    def generate_cauchy_greater_than_zero(self, mean: float, scale: float, min_val: float = None,
                                          max_val: float = None) -> float:
        """
        Generate a random value from a Cauchy distribution.

        Parameters:
        - mean (float): The location parameter (mean) of the Cauchy distribution.
        - scale (float): The scale parameter of the Cauchy distribution.
        - min_val (float, optional): Minimum value to clip the result.
        - max_val (float, optional): Maximum value to clip the result.

        Returns:
        - float: A random value from the Cauchy distribution.
        """
        while True:
            value = np.random.standard_cauchy() * scale + mean
            if value > 0:  # Ensure positive values
                break
        if min_val is not None or max_val is not None:
            value = np.clip(value, min_val, max_val)
        return value
