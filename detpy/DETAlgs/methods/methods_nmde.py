import random
import numpy as np
import copy

from detpy.DETAlgs.methods.methods_de import binomial_crossing_ind
from detpy.models.enums.optimization import OptimizationType
from detpy.models.member import Member
from detpy.models.population import Population


def nmde_mutation(population: Population, f_arr):
    new_members = []
    for i in range(population.size):
        pop_without_element = population.members.tolist()
        pop_without_element.pop(i)

        selected_members = random.sample(pop_without_element, 3)
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

def nmde_binomial_crossing(origin_population: Population, mutated_population: Population, cr_arr):
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


def nmde_selection(origin_population: Population, modified_population: Population):
    if origin_population.size != modified_population.size:
        print("Selection: populations have different sizes")
        return None

    if origin_population.optimization != modified_population.optimization:
        print("Selection: populations have different optimization types")
        return None

    optimization = origin_population.optimization
    new_members = []
    better_members_indexes = []
    for i in range(origin_population.size):
        if optimization == OptimizationType.MINIMIZATION:
            if origin_population.members[i] <= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))
                better_members_indexes.append(i)
        elif optimization == OptimizationType.MAXIMIZATION:
            if origin_population.members[i] >= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))
                better_members_indexes.append(i)

    new_population = Population(
        lb=origin_population.lb,
        ub=origin_population.ub,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population, better_members_indexes


def nmde_calculate_fm_crm(set_f: set, set_cr: set):
    list_f, list_cr = list(set_f), list(set_cr)
    f_m = sum(list_f) / len(list_f) if len(list_f) > 0.0 else 0.0
    cr_m = sum(list_cr) / len(list_cr) if len(list_cr) > 0.0 else 0.0
    return f_m, cr_m


def nmde_update_f_cr(f_m, cr_m, delta_f, delta_cr):
    f_i = np.random.uniform(low=f_m - delta_f, high=f_m + delta_f)
    cr_i = np.random.uniform(low=cr_m - delta_cr, high=cr_m + delta_cr)

    if f_i > 2:
        f_i = 2
    elif f_i < 0:
        f_i = 0.2

    if cr_i > 1:
        cr_i = 1
    elif cr_i < 0:
        cr_i = 0.1

    return f_i, cr_i
