import copy

import numpy as np
from numpy import ndarray

from detpy.models.enums.optimization import OptimizationType
from detpy.models.member import Member
from detpy.models.population import Population

def calculate_epsilon_constrained(population : Population, g_funcs: list, h_funcs: list, penalty_power: int) -> list:
    return list(epsilon_constrained_method(member.get_chromosomes(), g_funcs, h_funcs, penalty_power) for member in population.members)

def epsilon_constrained_method(chromosomes: ndarray, g_funcs: list, h_funcs: list, penalty_power: int) -> float:
    """
          Formula: sum(max(0, g_j(x))^p) + sum(h_j(x)^p)
    """
    g_constraint_violation = sum((max(0, pow(g(chromosomes), penalty_power))  for g in g_funcs))
    h_constraint_violation = sum((pow(h(chromosomes), penalty_power)  for h in h_funcs))
    return g_constraint_violation + h_constraint_violation

def epsilon_level_comparisons(reference_member: Member, comparison_member: Member,
                              origin_epsilon_constrained : float, modified_epsilon_constrained : float,
                              epsilon_level : int, optimization : OptimizationType) -> bool:
    if ((origin_epsilon_constrained <= epsilon_level and modified_epsilon_constrained <= epsilon_level)
            or origin_epsilon_constrained == modified_epsilon_constrained):
        if optimization == OptimizationType.MINIMIZATION:
            if reference_member <= comparison_member:
                return True
            else:
                return False
        elif optimization == OptimizationType.MAXIMIZATION:
            if reference_member >= comparison_member:
                return True
            else:
                return False
    else:
        if origin_epsilon_constrained < modified_epsilon_constrained:
            return True
        else:
            return False

def selection(origin_population: Population, modified_population: Population,
              origin_epsilon_constrained : list[float], modified_epsilon_constrained : list[float], selected : list[float],
              epsilon_level : int) -> Population:
    """
       Perform selection operation for the population.
       Parameters:
       - origin_population (Population): The original population.
       - modified_population (Population): The modified population.
       - origin_epsilon_constrained (list[float]): The original population epsilon constrained population.
       - modified_epsilon_constrained (list[float]): The modified population epsilon constrained population.
       - epsilon_level (int): The epsilon level.

       Returns: A new population with the selected chromosomes.
   """

    if origin_population.size != modified_population.size:
        print("Selection: populations have different sizes")
        return None

    if origin_population.optimization != modified_population.optimization:
        print("Selection: populations have different optimization types")
        return None

    optimization = origin_population.optimization
    new_members = []
    for i in range(origin_population.size):
        if epsilon_level_comparisons(origin_population.members[i], modified_population.members[i], origin_epsilon_constrained[i], modified_epsilon_constrained[i], epsilon_level, optimization):
            new_members.append(copy.deepcopy(origin_population.members[i]))
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