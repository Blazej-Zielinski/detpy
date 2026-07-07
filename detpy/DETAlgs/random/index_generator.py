import numpy as np


class IndexGenerator:
    """
    A general class to generate indices from a given range.
    """

    def generate_unique_multiple(self, total_size, count, exclude_indices):
        """
        Generate multiple unique random indices from a given range while avoiding the specified indices.

        Parameters:
        - total_size (int): The total number of indices available (e.g., range(0, total_size)).
        - count (int): The number of unique indices to generate.
        - exclude_indices (list[int]): A list of indices to exclude.

        Returns:
        - list[int]: A list of randomly selected unique indices that are not in the exclude_indices list.
        """
        candidates = [idx for idx in range(total_size) if idx not in exclude_indices]
        if len(candidates) < count:
            raise ValueError("Not enough valid indices available to select the requested number of unique indices.")
        return list(np.random.choice(candidates, count, replace=False))

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