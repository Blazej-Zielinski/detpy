
import numpy as np

from detpy.DETAlgs.methods.methods_de import mutation_ind, binomial_crossing_ind, \
    exponential_crossing_ind
from detpy.models.enums.crossingtype import CrossingType
from detpy.models.population import Population


def mutation(
        population: Population,
        f: list[float],
        ranks : list[int],
):
    new_members = []
    for i in range(population.size):
        r1, r2, r3 = ranks[i]
        new_member = mutation_ind(population.members[r1], [population.members[r2], population.members[r3]], f[i])
        new_members.append(new_member)

    new_population = Population(
        lb=population.lb,
        ub=population.ub,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population


def crossing(origin_population: Population, mutated_population: Population, cr: list[float],
             crossing_type: CrossingType):
    if origin_population.size != mutated_population.size:
        print("Populations have different sizes")
        return None

    new_members = []
    for i in range(origin_population.size):
        if crossing_type == CrossingType.BINOMIAL:
            new_member = binomial_crossing_ind(origin_population.members[i], mutated_population.members[i], cr[i])
        elif crossing_type == CrossingType.EXPOTENTIAL:
            new_member = exponential_crossing_ind(origin_population.members[i], mutated_population.members[i], cr[i])
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

def calculate_mutation_factors(population_siz : int, ranks : list[int], index_ranks: list[int], min_mutation_factor : float, max_mutation_factor : float):
        return list(min_mutation_factor + (max_mutation_factor - min_mutation_factor) * ((ranks[index_ranks[i][0]] - 1) / (population_siz - 1)) for i in range(population_siz))

def calculate_crossover_rates(population_siz : int, ranks : list[int], index_ranks: list[int],  min_crossover_rate : float, max_crossover_rate : float):
    return list(max_crossover_rate - (max_crossover_rate - min_crossover_rate) * ((ranks[index_ranks[i][0]] - 1) / (population_siz - 1)) for i in range(population_siz))

def create_ranks(epsilon_constrained):
    n_constraints = len(epsilon_constrained)
    order = sorted(range(n_constraints), key=lambda i: epsilon_constrained[i])

    ranks = [0] * n_constraints
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank

    return ranks