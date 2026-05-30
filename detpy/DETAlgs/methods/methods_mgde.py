import copy
import random
import numpy as np

from detpy.models.enums.mutation import mutation_rand_1
from detpy.models.member import Member
from detpy.models.population import Population


def mgde_mutation(population: Population, curr_generation: int, max_generation: float, mutation_factor_f: float,
                  mutation_factor_k: float) -> Population:
    pop_members_list = population.members.tolist()
    best_member = population.get_best_members(1)[0]

    new_members = []
    for i, member in enumerate(population.members):
        indices = list(range(population.size))
        indices.remove(i)
        selected_idxs = random.sample(indices, 3)

        donors = [pop_members_list[idx] for idx in selected_idxs]
        if random.random() > (curr_generation - 1) / (max_generation - 1):
            new_members.append(mutation_rand_1(donors, mutation_factor_f))
        else:
            new_members.append(mutation_curr_to_best_1_modified(member, best_member, donors,
                                                                mutation_factor_f, mutation_factor_k))

    new_population = Population(
        lb=population.lb,
        ub=population.ub,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population


def mgde_adapt_threshold(population: Population, threshold: float, mu: float, func):
    sorted_members = population.get_best_members(population.size)
    best_member = sorted_members[0]

    if best_member.fitness_value < threshold:
        threshold /= 10
        if random.random() > 0.5:
            worst_members = sorted_members[-5:]
            worst_member = worst_members[random.randint(0, 4)]
            member_chromosome_values = np.array([chromosome.real_value for chromosome in worst_member.chromosomes])
            new_chromosome_values = mu * member_chromosome_values * (1 - member_chromosome_values)
            for i in range(worst_member.args_num):
                worst_member.chromosomes[i].real_value = new_chromosome_values[i]
        else:
            member = sorted_members[random.randint(1, population.size - 1)]
            d = random.randint(0, member.args_num - 1)
            n = random.randint(0, 1)
            h = np.random.normal(loc=0.5, scale=0.3)

            temp = copy.deepcopy(member)
            temp.chromosomes[d] = member.chromosomes[d] + (
                    member.chromosomes[d] - best_member.chromosomes[d]) * h * (-1) ** n

            temp.calculate_fitness_fun(func)
            if temp < best_member:
                sorted_members[0] = temp


def mutation_curr_to_best_1_modified(base_member: Member, best_member: Member, members: list[Member], f: float,
                                     k: float) -> Member:
    """
        Formula: v_ij = x_r1 + F(x_r2 - x_r3) + K(x_best - x_base)
    """
    new_member = copy.deepcopy(base_member)
    new_member.chromosomes = members[0].chromosomes + (members[1].chromosomes - members[2].chromosomes) * f + (
            best_member.chromosomes - base_member.chromosomes) * k
    return new_member
