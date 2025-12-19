from typing import List

import numpy as np

from detpy.models.enums.optimization import OptimizationType
from detpy.models.member import Member


def archive_reduction(archive: list[Member], archive_size: int,
                      optimization: OptimizationType = OptimizationType.MINIMIZATION):
    """
    Reduce the size of the archive to the specified size by removing the worst members.

    Parameters:
    - archive (list[Member]): The archive to reduce.
    - archive_size (int): The desired size of the archive.
    - optimization (OptimizationType): The optimization type (MINIMIZATION or MAXIMIZATION).
    """
    if archive_size == 0:
        archive.clear()
        return

    if len(archive) > archive_size:
        # Sort archive by fitness based on the optimization type
        archive.sort(
            key=lambda member: member.fitness_value,
            reverse=(optimization == OptimizationType.MAXIMIZATION)  # Reverse for maximization problems
        )
        # Remove the worst members
        del archive[archive_size:]


def rank_selection(members: List[Member], k: float = 1.0,
                   optimization: OptimizationType = OptimizationType.MINIMIZATION) -> (int, int):
    """
    Choose two random members from the population using rank selection.

    Parameters:
    - members (Population): The population from which to select the members.
    - k (float): Scaling factor for rank selection.
    - optimization (OptimizationType): The optimization type (MINIMIZATION or MAXIMIZATION).

    Returns: The indexes of the selected members.
    """
    # Sort members based on fitness value (ascending for minimization, descending for maximization)
    sorted_population_member_indexes = np.argsort(
        [member.fitness_value for member in members]
    )
    if optimization == OptimizationType.MAXIMIZATION:
        sorted_population_member_indexes = sorted_population_member_indexes[::-1]

    # Assign ranks and calculate probabilities
    values = list(range(len(members), 0, -1))
    ranks = k * np.array(values)
    probabilities = ranks / ranks.sum()

    # Ensure distinct indices
    selected_indices = np.random.choice(len(sorted_population_member_indexes), size=2, replace=False, p=probabilities)
    selected_values = sorted_population_member_indexes[selected_indices]

    return int(selected_values[0]), int(selected_values[1])
