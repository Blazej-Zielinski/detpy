import copy
import random

import numpy as np

from detpy.DETAlgs.methods.methods_eps_de import epsilon_constrained_method
from detpy.DETAlgs.methods.methods_eps_deag import calculate_delta_x
from detpy.models.enums.boundary_constrain import boundary_clipping
from detpy.models.enums.derivative_method import DerivativeMethod
from detpy.models.population import Population


def gradient_mutation(pop_population: Population, number_of_repeating_mutation: int, gradient_base_mutation_rate: float,
                      epsilon_level: float, derivative_method: DerivativeMethod, g_funcs: list, h_funcs: list,
                      penalty_power: int, tolerance_h: float):
    new_members = []
    for i in range(pop_population.size):
        member = copy.deepcopy(pop_population.members[i])
        epsilon_constrain = epsilon_constrained_method(member.get_chromosomes(), g_funcs, h_funcs, penalty_power,
                                                       tolerance_h)
        if epsilon_constrain > epsilon_level and random.uniform(0, 1) < gradient_base_mutation_rate:
            for _ in range(number_of_repeating_mutation):
                delta_x = calculate_delta_x(member.get_chromosomes(), derivative_method, g_funcs, h_funcs,
                                            epsilon_constrain)

                for j in range(member.args_num):
                    member.chromosomes[j].real_value = member.chromosomes[j].real_value + delta_x.item(j)

                boundary_clipping(member)

                if epsilon_constrained_method(member.get_chromosomes(), g_funcs, h_funcs, penalty_power,
                                              tolerance_h) <= epsilon_level:
                    break

        new_members.append(member)

    new_population = Population(
        lb=pop_population.lb,
        ub=pop_population.ub,
        arg_num=pop_population.arg_num,
        size=pop_population.size,
        optimization=pop_population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population


def calculate_init_epsilon_level(epsilon_constrained: list, theta: int):
    epsilon_constrained_length = len(epsilon_constrained)
    if theta >= epsilon_constrained_length or theta < 0:
        raise ValueError(f"Theta must be within the range 0 to {epsilon_constrained_length}, current value: {theta}")
    return sorted(epsilon_constrained)[theta]
