import copy
import random
import numpy as np

from detpy.models.member import Member
from detpy.models.population import Population


def derl_mutation(population: Population):
    new_members = []
    for _ in range(population.size):
        selected_members = np.array(random.sample(population.members.tolist(), 3))
        sorted_indices = np.argsort([member.fitness_value for member in selected_members])
        best_member, better_member, worst_member = selected_members[sorted_indices]

        f = random.uniform(-0.6, -0.4) if random.random() < 0.5 else random.uniform(0.4, 0.6)  # (−1, -0.4) ∪ (0.4, 1)
        new_member = mutation_ind(best_member, better_member, worst_member, f)

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
