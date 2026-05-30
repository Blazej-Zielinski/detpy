import numpy as np
import math

from detpy.DETAlgs.methods.methods_de import binomial_crossing_ind
from detpy.models.population import Population


def ide_binomial_crossing(origin_population: Population, mutated_population: Population, cr_arr):
    if origin_population.size != mutated_population.size:
        print("Binomial_crossing: populations have different sizes")
        return None

    new_members = []
    for i in range(origin_population.size):
        new_member = binomial_crossing_ind(origin_population.members[i], mutated_population.members[i], cr_arr[i])
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


def ide_get_f(curr_gen, max_gen):
    f_0 = 0.5
    exponent = 1 - (max_gen / (max_gen + 1 - curr_gen))
    return f_0 * 2 ** (math.e ** exponent)


def ide_get_cr(pop: Population):
    cr_arr = np.zeros(pop.size)
    cr_min, cr_max = 0.3, 0.9

    fitness_max = pop.get_best_members(pop.size)[-1].fitness_value
    fitness_mean = pop.mean()

    for i in range(pop.size):
        member_fitness = pop.members[i].fitness_value

        if member_fitness > fitness_mean:
            cr_arr[i] = cr_min + (cr_max - cr_min) * ((member_fitness - fitness_mean) / (fitness_max - fitness_mean))
        else:
            cr_arr[i] = cr_min

    return cr_arr
