import copy
from typing import List

import numpy as np

from detpy.models.member import Member
from detpy.models.population import Population


def archive_reduction(archive: list[Member], archive_size: int):
    """
    Reduce the size of the archive to the specified size by removing the worst members.

    Parameters:
    - archive (list[Member]): The archive to reduce.
    - archive_size (int): The desired size of the archive.
    """
    if archive_size == 0:
        archive.clear()
        return

    if len(archive) > archive_size:
        # Sort archive by fitness (worst first)
        archive.sort(key=lambda member: member.fitness_value, reverse=True)
        # Remove the worst members
        del archive[archive_size:]


def mutation_internal(base_member: Member, best_member: Member, r1: Member, r2: Member, f: float, fw: float):
    """
    Formula: bm + Fw * (bm_best - bm) + F * (r1 - r2)

    Parameters:
    - base_member (Member): The base member used for the mutation operation.
    - best_member (Member): The best member, typically the one with the best fitness value, used in the mutation formula.
    - r1 (Member): A randomly selected member from the population, used for mutation. (rank selection)
    - r2 (Member): Another randomly selected member, used for mutation. (rank selection)
    - f (float): A scaling factor that controls the magnitude of the mutation between random members of the population.
    - fw (float): A scaling factor that controls the magnitude of the mutation between the best member and the base member.

    Returns: A new member with the mutated chromosomes.
    """
    new_member = copy.deepcopy(base_member)
    new_member.chromosomes = base_member.chromosomes + (
            fw * (best_member.chromosomes - base_member.chromosomes)) + (
                                     f * (r1.chromosomes - r2.chromosomes))
    return new_member


def crossing_internal(org_member: Member, mut_member: Member, cr: float):
    """
    Perform crossing operation for two members.

    Parameters:
    - org_member (Member): The original member.
    - mut_member (Member): The mutated member.
    - cr (float): The crossover rate.

    Returns: A new member with the crossed chromosomes.
    """
    new_member = copy.deepcopy(org_member)

    random_numbers = np.random.rand(new_member.args_num)
    mask = random_numbers <= cr

    # ensures that new member gets at least one parameter
    i_rand = np.random.randint(low=0, high=new_member.args_num)

    for i in range(new_member.args_num):
        if mask[i] or i_rand == i:
            new_member.chromosomes[i].real_value = mut_member.chromosomes[i].real_value
        else:
            new_member.chromosomes[i].real_value = org_member.chromosomes[i].real_value

    return new_member


def crossing(origin_population: Population, mutated_population: Population, cr_table):
    """
    Perform crossing operation for the population.
    Parameters:
    - origin_population (Population): The original population.
    - mutated_population (Population): The mutated population.
    - cr (float): The crossover rate.

    Returns: A new population with the crossed chromosomes.
    """
    if origin_population.size != mutated_population.size:
        print("Binomial_crossing: populations have different sizes")
        return None

    new_members = []
    for i in range(origin_population.size):
        new_member = crossing_internal(origin_population.members[i], mutated_population.members[i], cr_table[i])
        new_members.append(new_member)

    new_population = Population(
        lb=origin_population.lb,
        ub=origin_population.ub,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population


def rank_selection(members: List[Member], k: float = 1.0) -> (int, int):
    """
    Choose two random members from the population using rank selection.
    Parameters:
    - members (Population): The population from which to select the members.
    - k (float): Scaling factor for rank selection.

    Returns: The indexes of the selected members.
    """
    sorted_population_member_indexes = np.argsort([member.fitness_value for member in members])
    values = list(range(len(members), 0, -1))
    ranks = k * np.array(values)
    probabilities = ranks / ranks.sum()

    # Ensure distinct indices
    selected_indices = np.random.choice(len(sorted_population_member_indexes), size=2, replace=False, p=probabilities)
    selected_values = sorted_population_member_indexes[selected_indices]

    return int(selected_values[0]), int(selected_values[1])
