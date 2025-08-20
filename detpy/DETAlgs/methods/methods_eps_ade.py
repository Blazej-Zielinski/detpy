import random

import numpy as np

from detpy.DETAlgs.methods.methods_de import mutation_ind, binomial_crossing_ind
from detpy.models.enums.boundary_constrain import boundary_clipping
from detpy.models.fitness_function import FitnessFunctionBase
from detpy.models.member import Member
from detpy.models.population import Population


def control_epsilon_level(epsilon_level: float, epsilon_constrained: list[float],
                          truncation_mechanism_factory: float, population_size: int):
    if epsilon_constrained.count(0) > truncation_mechanism_factory * population_size:
        return 0
    else:
        clip_min = truncation_mechanism_factory * np.min(epsilon_constrained)
        clip_max = truncation_mechanism_factory * np.max(epsilon_constrained)
        return np.clip(epsilon_level, clip_min, clip_max)


def adaptive_de_operation(origin_population: Population, origin_member : Member, mutation_factor : float, crossover_rate : float, fitness_function_base : FitnessFunctionBase) -> Member:

    diff_members = random.sample(origin_population.members.tolist(), 3)
    mutation_member = mutation_ind(diff_members[0], diff_members[1:], mutation_factor)
    boundary_clipping(mutation_member)
    new_member = binomial_crossing_ind(origin_member, mutation_member, crossover_rate)
    Population.calculate_fitness(new_member, fitness_function_base.eval)
    return new_member