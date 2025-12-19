import copy

import numpy as np

from detpy.models.member import Member
from random import randrange


def calculate_best_member_count(population_size: int, p_best_fraction: float = None) -> int:
    """
    Calculate the number of the best members to select.

    If p_best_fraction is provided, use it directly.
    Otherwise, choose a random percentage between 2/population_size and 20%.

    Parameters:
    - population_size (int): Size of the population.
    - p_best_fraction (float, optional): Fraction of best individuals to select.

    Returns:
    - int: Number of best individuals to select.
    """

    if p_best_fraction is not None:
        return max(2, int(p_best_fraction * population_size))  # min 2 individuals
    else:
        min_percentage = 2 / population_size
        max_percentage = 0.2
        random_percentage = np.random.uniform(min_percentage, max_percentage)
        return int(random_percentage * population_size)


def archive_reduction(archive: list, pop_size: int, w_ext: float) -> list:
    """
    Reduce the size of the archive based on w_ext * pop_size.

    Parameters:
    - archive (list): The archive of members from previous populations.
    - pop_size (int): The current population size.
    - w_ext (float): Weight determining archive size as a multiple of population size.

    Returns:
    - list: The reduced archive.
    """
    expected_size = int(w_ext * pop_size)

    # If the expected size is zero, clear the archive
    if expected_size == 0:
        archive.clear()
        return archive

    # Reduce archive if necessary
    while len(archive) > expected_size:
        idx = randrange(len(archive))
        archive.pop(idx)

    return archive


def mutation_internal(base_member: Member, best_member: Member, r1: Member, r2: Member, f: float):
    """
    Formula: bm + Fw * (bm_best - bm) + F * (r1 - r2)

    Parameters:
    - base_member (Member): The base member used for the mutation operation.
    - best_member (Member): The best member, typically the one with the best fitness value, used in the mutation formula.
    - r1 (Member): A randomly selected member from the population, used for mutation. (rank selection)
    - r2 (Member): Another randomly selected member from archive, used for mutation
    - f (float): A scaling factor that controls the magnitude of the mutation between random members of the population.

    Returns: A new member with the mutated chromosomes.
    """
    new_member = copy.deepcopy(base_member)
    new_member.chromosomes = base_member.chromosomes + (
            f * (best_member.chromosomes - base_member.chromosomes)) + (
                                     f * (r1.chromosomes - r2.chromosomes))
    return new_member


