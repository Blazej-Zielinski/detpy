import numpy as np


class IndexGenerator:
    """
    A general class to generate indices from a given range.
    """

    def generate_unique(self, total_size, exclude_indices):
        """
        Generate a random index from a given range while avoiding the specified indices.

        Parameters:
        - total_size (int): The total number of indices available (e.g., range(0, total_size)).
        - exclude_indices (list[int]): A list of indices to exclude.

        Returns:
        - int: A randomly selected index that is not in the exclude_indices list.
        """
        candidates = [idx for idx in range(total_size) if idx not in exclude_indices]
        if not candidates:
            raise ValueError("No valid indices available to select from.")
        return np.random.choice(candidates, 1, replace=False)[0]

    def generate(self, min_value, max_value):
        """
        Generate a random index from a given range [min_value, max_value).

        Parameters:
        - min_value (int): The minimum value of the range (inclusive).
        - max_value (int): The maximum value of the range (exclusive).

        Returns:
        - int: A randomly selected index.
        """
        return np.random.randint(min_value, max_value)