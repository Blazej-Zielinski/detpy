import random
import numpy as np
import copy

from detpy.models.member import Member
from detpy.models.population import Population
from detpy.models.enums.optimization import OptimizationType


def mutation_ind(base_member: Member, member1: Member, member2: Member, fs):
    """
        Formula: v_ij = x_r1 + FS(x_r2 - x_r3)
    """
    new_member = copy.deepcopy(base_member)
    new_member.chromosomes = base_member.chromosomes + (member1.chromosomes - member2.chromosomes) * fs
    return new_member


def mutation(population: Population, fs):
    new_members = []
    for _ in range(population.size):
        selected_members = random.sample(population.members.tolist(), 3)
        new_member = mutation_ind(selected_members[0], selected_members[1], selected_members[2], fs)
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


def binomial_crossing_ind(org_member: Member, mut_member: Member, cr):
    new_member = copy.deepcopy(org_member)

    random_numbers = np.random.rand(new_member.args_num)
    mask = random_numbers <= cr
    i_rand = np.random.randint(low=0, high=new_member.args_num)

    for i in range(new_member.args_num):
        if mask[i] or i_rand == i:
            new_member.chromosomes[i].real_value = mut_member.chromosomes[i].real_value
        else:
            new_member.chromosomes[i].real_value = org_member.chromosomes[i].real_value

    return new_member


def improve_de_binomial_crossing(origin_population: Population, mutated_population: Population, cr):
    if origin_population.size != mutated_population.size:
        print("Binomial_crossing: populations have different sizes")
        return None

    new_members = []
    for i in range(origin_population.size):
        new_member = binomial_crossing_ind(origin_population.members[i], mutated_population.members[i], cr)
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


def selection(origin_population: Population, modified_population: Population):
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
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))
        elif optimization == OptimizationType.MAXIMIZATION:
            if origin_population.members[i] >= modified_population.members[i]:
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
