import copy
from random import randrange

import numpy as np

from detpy.models.chromosome import Chromosome
from detpy.models.member import Member
from detpy.models.population import Population


def wrap_as_chromosomes(float_array: np.ndarray, template_chromosomes: list) -> list:
    chromosomes = []
    for val, template in zip(float_array, template_chromosomes):
        c = Chromosome(template.lb, template.ub)
        c.real_value = val
        chromosomes.append(c)
    return chromosomes


def calculate_best_member_count(population_size: int):
    """
    Calculate the number of the best members to select based on a percentage of the population size.
    The percentage is randomly chosen between a minimum value (2/population_size) and a maximum value (20% of the population).

    Parameters:
    - population_size (int): The size of the population.

    Returns:
    - int: The number of the best members to select.
    """

    min_percentage = 2 / population_size
    max_percentage = 0.2

    random_percentage = np.random.uniform(min_percentage, max_percentage)

    return int(random_percentage * population_size)


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

    i_rand = np.random.randint(low=0, high=new_member.args_num)

    for i in range(new_member.args_num):
        if mask[i] or i_rand == i:
            new_member.chromosomes[i].real_value = mut_member.chromosomes[i].real_value
        else:
            new_member.chromosomes[i].real_value = org_member.chromosomes[i].real_value

    return new_member
