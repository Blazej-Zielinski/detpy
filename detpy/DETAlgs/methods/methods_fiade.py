import numpy as np
import random
import copy
from detpy.models.enums.mutation import mutation_rand_1
from detpy.models.enums.optimization import OptimizationType
from detpy.models.population import Population
from detpy.DETAlgs.methods.methods_de import binomial_crossing_ind

def mutation(population: Population, mutation_factors: list[list[float, bool]]) -> Population:
    members = population.members.tolist()
    drew_members = [random.sample(members, 3) for _ in range(population.size)]

    new_members = [
        mutation_rand_1(selected_members, f)
        for selected_members, (f, _) in zip(drew_members, mutation_factors)
    ]

    new_population = Population(
        lb=population.lb,
        ub=population.ub,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population


def fiade_crossing(origin_population: Population, mutated_population: Population,
             crossover_rates: list[list[float, bool]]) -> Population:
    if origin_population.size != mutated_population.size:
        raise ValueError("Populations must have the same size for crossing")

    new_members = [
        binomial_crossing_ind(origin_population.members[i], mutated_population.members[i], crossover_rates[i][0])
        for i in range(origin_population.size)
    ]

    new_population = Population(
        lb=origin_population.lb,
        ub=origin_population.ub,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population


def selection(origin_population: Population, modified_population: Population, mutation_factors: list[list[float, bool]],
              crossover_rates: list[list[float, bool]]) -> Population:
    if origin_population.size != modified_population.size:
        raise ValueError("Populations must have the same size for selection")

    if origin_population.optimization != modified_population.optimization:
        raise ValueError("Populations must have the same optimization type")

    optimization = origin_population.optimization
    new_members = []

    for i in range(origin_population.size):
        if optimization == OptimizationType.MINIMIZATION:
            if origin_population.members[i] <= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
                mutation_factors[i] = update_mutation_factor(origin_population.members[i],
                                                             modified_population.members[i], mutation_factors[i])
                crossover_rates[i] = update_crossover_rate(origin_population.members[i], modified_population.members[i],
                                                           crossover_rates[i])
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))
        else:  # MAXIMIZATION
            if origin_population.members[i] >= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
                mutation_factors[i] = update_mutation_factor(origin_population.members[i],
                                                             modified_population.members[i], mutation_factors[i])
                crossover_rates[i] = update_crossover_rate(origin_population.members[i], modified_population.members[i],
                                                           crossover_rates[i])
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))

    new_population = Population(
        lb=origin_population.lb,
        ub=origin_population.ub,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population


def adapt_parameters(population: Population, mutation_factors: list[list[float, bool]],
                     crossover_rates: list[list[float, bool]]):
    sorted_members = population.get_best_members(population.size)
    max_fitness_value = sorted_members[-1].fitness_value
    min_fitness_value = sorted_members[0].fitness_value

    for i in range(population.size):
        if mutation_factors[i][1]:
            mutation_factors[i][1] = False
        else:
            mutation_factors[i] = [random.random() * ((max_fitness_value - min_fitness_value) / max_fitness_value),
                                   False]

        if crossover_rates[i][1]:
            crossover_rates[i][1] = False
        else:
            crossover_rates[i] = [random.random() * ((max_fitness_value - min_fitness_value) / max_fitness_value),
                                  False]


def update_mutation_factor(orig_member, mod_member, mutation_factor):
    try:
        delta = (mod_member.fitness_value - orig_member.fitness_value) / mod_member.fitness_value * random.random()
        return [mutation_factor[0] - delta, True]
    except ZeroDivisionError:
        return [0, True]


def update_crossover_rate(orig_member, mod_member, crossover_rate):
    try:
        delta = (mod_member.fitness_value - orig_member.fitness_value) / mod_member.fitness_value * random.random()
        return [crossover_rate[0] - delta, True]
    except ZeroDivisionError:
        return [0, True]