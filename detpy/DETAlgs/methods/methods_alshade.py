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


def fix_boundary_constraints(population: Population, trial: Population):
    for member, member_pop in zip(trial.members, population.members):
        for chromosome, chromosome_pop in zip(member.chromosomes, member_pop.chromosomes):
            if chromosome.real_value > chromosome.ub:
                chromosome.real_value = (chromosome.ub + chromosome_pop.real_value) / 2
            elif chromosome.real_value < chromosome.lb:
                chromosome.real_value = (chromosome.lb + chromosome_pop.real_value) / 2


def current_to_xamean(base_member: Member, xamean: Member, r1: Member, r2: Member, f: float) -> Member:
    """
    Implements the current-to-Amean/1 mutation strategy:

        v_i = x_i + F * (x_Amean - x_i) + F * (x_r1 - x_r2)
    """
    xamean_chromosomes = wrap_as_chromosomes(xamean, base_member.chromosomes)
    xamean_as_member = copy.deepcopy(base_member)
    xamean_as_member.chromosomes = xamean_chromosomes

    new_member = copy.deepcopy(base_member)
    new_member.chromosomes = (
            base_member.chromosomes
            + f * (xamean_as_member.chromosomes - base_member.chromosomes)
            + f * (r1.chromosomes - r2.chromosomes)
    )
    return new_member


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
