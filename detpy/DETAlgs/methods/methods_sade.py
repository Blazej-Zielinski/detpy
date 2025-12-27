import copy
import random
import numpy as np
from detpy.DETAlgs.methods.methods_de import binomial_crossing_ind
from detpy.models.enums.optimization import OptimizationType
from detpy.models.member import Member
from detpy.models.population import Population


def sade_mutation(population: Population, f_arr):
    new_members = []
    for i in range(population.size):
        selected_members = random.sample(population.members.tolist(), 3)
        new_member = mutation_ind(selected_members[0], selected_members[1], selected_members[2], f_arr[i])
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


def mutation_ind(base_member: Member, member1: Member, member2: Member, f):
    """
        Formula: v_ij = x_r1 + F(x_r2 - x_r3)
    """
    new_member = copy.deepcopy(base_member)
    new_member.chromosomes = base_member.chromosomes + (member1.chromosomes - member2.chromosomes) * f
    return new_member


def sade_binomial_crossing(origin_population: Population, mutated_population: Population, cr_arr):
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


def sade_selection(origin_population: Population, modified_population: Population, f_arr, cr_arr, prob_f, prob_cr):
    if origin_population.size != modified_population.size:
        print("Selection: populations have different sizes")
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
                if np.random.uniform() < prob_f:
                    f_arr[i] = np.random.uniform()
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))
                if np.random.uniform() >= prob_cr:
                    cr_arr[i] = np.random.uniform()
        elif optimization == OptimizationType.MAXIMIZATION:
            if origin_population.members[i] >= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))
                f_arr[i] = np.random.uniform() if np.random.uniform() < prob_f else f_arr[i]
                cr_arr[i] = np.random.uniform() if np.random.uniform() < prob_cr else cr_arr[i]

    new_population = Population(
        lb=origin_population.lb,
        ub=origin_population.ub,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population, f_arr, cr_arr
