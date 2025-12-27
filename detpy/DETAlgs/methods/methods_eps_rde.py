import random

import numpy as np

from detpy.DETAlgs.methods.methods_de import get_best_member, mutation_ind, binomial_crossing_ind, \
    exponential_crossing_ind
from detpy.models.enums.basevectorschema import BaseVectorSchema
from detpy.models.enums.crossingtype import CrossingType
from detpy.models.enums.optimization import OptimizationType
from detpy.models.population import Population


def mutation(
        population: Population,
        base_vector_schema: BaseVectorSchema,
        optimization_type: OptimizationType,
        f: list[float],
        y: int
):
    new_members = []
    best_member = get_best_member(optimization_type, population)

    for i in range(population.size):
        diff_members = random.sample(population.members.tolist(), 2 * y)

        if base_vector_schema == BaseVectorSchema.RAND:
            base_vector = random.choice(population.members.tolist())
        elif base_vector_schema == BaseVectorSchema.CURRENT:
            base_vector = population.members[i]
        elif base_vector_schema == BaseVectorSchema.BEST:
            base_vector = best_member
        else:
            raise ValueError("Unknown base vector schema.")

        new_member = mutation_ind(base_vector, diff_members, f[i])
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


def calculate_mutation_factors(population: Population, ranks: list[int], min_mutation_factor: float,
                               max_mutation_factor: float):
    return list(
        min_mutation_factor + (max_mutation_factor - min_mutation_factor) * ((ranks[i] - 1) / (population.size - 1)) for
        i in range(population.size))


def calculate_crossover_rates(population: Population, ranks: list[int], min_crossover_rate: float,
                              max_crossover_rate: float):
    return list(
        max_crossover_rate - (max_crossover_rate - min_crossover_rate) * ((ranks[i] - 1) / (population.size - 1)) for i
        in range(population.size))


def create_ranks(population: Population, epsilon_constrained: list):
    return sorted(range(population.size), key=lambda i: epsilon_constrained[i])
