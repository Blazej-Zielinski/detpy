import copy
import random
import numpy as np

from detpy.models.member import Member
from detpy.models.population import Population


def comde_mutation(population: Population):
    new_members = []
    for _ in range(population.size):
        f_l = np.random.uniform()  # random in (0, 1]
        f_g = np.random.uniform(low=-1, high=1)  # random in (−1, 0) ∪ (0, 1)
        while f_g == 0 or f_l == 0:  # not pleasant
            f_l = np.random.uniform()
            f_g = np.random.uniform(low=-1, high=1)

        if np.random.uniform() <= 0.5:
            # Select worst and best members
            sorted_members = population.get_best_members(population.size)
            best_member, worst_member = sorted_members[0], sorted_members[-1]
            selected_member = random.sample(sorted_members[1:-1].tolist(), 1)[0]
            new_member = mutation_ind(selected_member, best_member, worst_member, f_l)
        else:
            # Select members by random
            selected_members = random.sample(population.members.tolist(), 3)
            new_member = mutation_ind(selected_members[0], selected_members[1], selected_members[2], f_g)

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


def calculate_cr(curr_gen, max_gen, cr_min=0.5, cr_max=0.95, k=4):
    cr = cr_max + (cr_min - cr_max) * (1 - curr_gen / max_gen) ** k
    return cr
