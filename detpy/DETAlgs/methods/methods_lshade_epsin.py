import copy
from random import randrange

import numpy as np

from detpy.models.member import Member
from detpy.models.population import Population


def calculate_best_member_count(population_size: int, p: float = 0.2) -> int:
    """
    Calculate the number of best members to select based on the population size and a given proportion p.
    Args:
        population_size (int):  The size of the population.
        p (float):  The proportion of the population to select as best members (default is 0.2).

    Returns: The number of best members to select.

    """
    return int(p * population_size)

def archive_reduction(archive: list[Member], archive_size: int, pop_size: int):
    """
    Reduce the size of the archive to the specified size.

    Parameters:
    - archive (list[Member]): The archive of members from previous populations.
    - archive_size (int): The desired size of the archive.
    - pop_size (int): The size of the population.

    Returns: The reduced archive.
    """
    if archive_size == 0:
        archive.clear()

    max_elem = min(archive_size, pop_size)
    reduce_num = len(archive) - max_elem

    for _ in range(reduce_num):
        idx = randrange(len(archive))
        archive.pop(idx)

    return archive


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


def crossing(origin_population: Population, mutated_population: Population, cr_table: list[float]):
    """
    Perform crossing operation for the population.
    Parameters:
    - origin_population (Population): The original population.
    - mutated_population (Population): The mutated population.
    - cr_table (List[float]): List of crossover rates.

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
