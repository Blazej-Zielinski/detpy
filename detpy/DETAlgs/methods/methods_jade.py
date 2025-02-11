import numpy as np
from math import floor
import random
import copy

from detpy.math_functions.lehmer_mean import LehmerMean
from detpy.models.population import Population
from detpy.models.member import Member
from detpy.models.enums.mutation import mutation_curr_to_best_1
from detpy.models.enums.optimization import OptimizationType
from detpy.DETAlgs.methods.methods_de import binomial_crossing_ind

lehmer_mean_func = LehmerMean()


def jade_mutation(population: Population, mutation_factors: np.ndarray[float], p_best: float,
                  archive: list[Member] = []) -> Population:
    pop_members_list = population.members.tolist()
    possible_members_list = pop_members_list + archive if archive else pop_members_list

    sort_reversed = False if population.optimization == OptimizationType.MINIMIZATION else True
    sorted_members = sorted(possible_members_list, key=lambda x: x.fitness_value, reverse=sort_reversed)
    p_best_members = sorted_members[:floor(p_best * population.size)]

    new_members = []
    for (i, base_member) in enumerate(pop_members_list):
        best_member = random.choice(p_best_members)

        indices = list(range(population.size))
        indices.remove(i)
        idx_r1 = random.choice(indices)
        x_r1 = pop_members_list[idx_r1]

        indices = list(range(len(possible_members_list)))
        indices.remove(i)
        indices.remove(idx_r1)
        idx_r2 = random.choice(indices)
        x_r2 = possible_members_list[idx_r2]

        new_member = mutation_curr_to_best_1(base_member, best_member, [x_r1, x_r2], mutation_factors[i])
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


def jade_binomial_crossing(origin_population: Population, mutated_population: Population,
                           crossover_rates: np.ndarray[float]) -> Population | None:
    if origin_population.size != mutated_population.size:
        print("Binomial_crossing: populations have different sizes")
        return None

    new_members = []
    for i in range(origin_population.size):
        new_member = binomial_crossing_ind(origin_population.members[i], mutated_population.members[i],
                                           crossover_rates[i])
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


def jade_selection(origin_population: Population, modified_population: Population,
                   mutation_factors: np.ndarray[float], crossover_rates: np.ndarray[float],
                   success_mutation_factors: list[float], success_crossover_rates: list[float],
                   archive: list[Member] = []) -> Population | None:
    if origin_population.size != modified_population.size != len(mutation_factors) != len(crossover_rates):
        print("Selection: parameters lists have different sizes")
        return None

    if origin_population.optimization != modified_population.optimization:
        print("Selection: populations have different optimization types")
        return None

    optimization = origin_population.optimization
    new_members = []
    for i in range(origin_population.size):
        if optimization == OptimizationType.MINIMIZATION:
            if origin_population.members[i] <= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))
                archive.append(copy.deepcopy(origin_population.members[i]))
                success_mutation_factors.append(mutation_factors[i])
                success_crossover_rates.append(crossover_rates[i])

        elif optimization == OptimizationType.MAXIMIZATION:
            if origin_population.members[i] >= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))
                archive.append(copy.deepcopy(origin_population.members[i]))
                success_mutation_factors.append(mutation_factors[i])
                success_crossover_rates.append(crossover_rates[i])

    new_population = Population(
        lb=origin_population.lb,
        ub=origin_population.ub,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    new_population.members = np.array(new_members)

    return new_population


def jade_reduce_archive(archive: list[Member], archive_size: int):
    if archive_size == 0:
        archive.clear()

    reduce_num = len(archive) - archive_size
    for _ in range(reduce_num):
        idx = random.randrange(len(archive))
        archive.pop(idx)


def jade_adapt_crossover_rates(c: float, cr_mean: float, cr_std: float, cr_low: float, cr_high: float, size: int,
                               success_crossover_rates: list[float]) -> tuple[np.ndarray[float], float]:
    if success_crossover_rates:
        a_mean = sum(success_crossover_rates) / len(success_crossover_rates)
        cr_mean = (1 - c) * cr_mean + c * a_mean

    new_crossover_rates = draw_norm_dist_within_bounds(cr_mean, cr_std, cr_low, cr_high, size)
    success_crossover_rates.clear()
    return new_crossover_rates, cr_mean


def jade_adapt_mutation_factors(c: float, f_mean: float, f_std: float, size: int,
                                success_mutation_factors: list[float]) -> tuple[np.ndarray[float], float]:
    if success_mutation_factors:
        f_mean = (1 - c) * f_mean + c * lehmer_mean_func.evaluate(success_mutation_factors)

    new_mutation_factors = draw_cauchy_dist_within_bounds(f_mean, f_std, size)
    success_mutation_factors.clear()
    return new_mutation_factors, f_mean


def draw_norm_dist_within_bounds(mean: float, std: float, low: float, high: float, arr_size: int) -> np.ndarray[float]:
    """
    Draw numbers from normal distributions within bounds (low, high].
    """
    values = np.clip(np.random.normal(loc=mean, scale=std, size=arr_size), low, high)

    while len(values) < arr_size:
        new_values = np.clip(np.random.normal(loc=mean, scale=std, size=arr_size - len(values)), low, high)
        values = np.concatenate((values, new_values))

    return values[:arr_size]


def draw_cauchy_dist_within_bounds(mean: float, std: float, arr_size: int) -> np.ndarray[float]:
    values = []

    while len(values) < arr_size:
        value = mean + std * np.random.standard_cauchy()

        if value >= 1:
            values.append(1.0)
        elif 0 < value < 1:
            values.append(value)

    return np.array(values)
