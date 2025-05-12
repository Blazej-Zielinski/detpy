import random
import numpy as np
import copy

from detpy.models.enums.basevectorschema import BaseVectorSchema
from detpy.models.enums.crossingtype import CrossingType
from detpy.models.member import Member
from detpy.models.population import Population
from detpy.models.enums.optimization import OptimizationType


def mutation_ind(base_member: Member, member1: Member, member2: Member, f):
    """
        Formula: v_ij = x_r1 + F(x_r2 - x_r3)
    """
    new_member = copy.deepcopy(base_member)
    new_member.chromosomes = base_member.chromosomes + (member1.chromosomes - member2.chromosomes) * f
    return new_member


def mutation(population: Population, base_vector_schema: BaseVectorSchema, optimization_type: OptimizationType, f):
    new_members = []

    best_member = get_best_member(optimization_type, population)

    for i in range(population.size):
        selected_members = random.sample(population.members.tolist(), 2)

        if base_vector_schema == BaseVectorSchema.RAND:
            base_vector = random.choice(population.members.tolist())
        if base_vector_schema == BaseVectorSchema.CURRENT:
            base_vector = population.members[i]
        if base_vector_schema == BaseVectorSchema.BEST:
            base_vector = best_member

        new_member = mutation_ind(base_vector, selected_members[0], selected_members[1], f)
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


def get_best_member(optimization, population):
    if optimization == OptimizationType.MINIMIZATION:
        sorted_indices = np.argsort([member.fitness_value for member in population.members])
    else:
        sorted_indices = np.argsort([member.fitness_value for member in population.members])[::-1]
    best_member = population.members[sorted_indices[0]]
    return best_member


def binomial_crossing_ind(org_member: Member, mut_member: Member, cr):
    new_member = copy.deepcopy(org_member)

    random_numbers = np.random.rand(new_member.args_num)
    mask = random_numbers <= cr

    # ensures that new member gets at least one parameter (giga important line)
    i_rand = np.random.randint(low=0, high=new_member.args_num)

    for i in range(new_member.args_num):
        if mask[i] or i_rand == i:
            new_member.chromosomes[i].real_value = mut_member.chromosomes[i].real_value
        else:
            new_member.chromosomes[i].real_value = org_member.chromosomes[i].real_value

    return new_member


def exponential_crossing_ind(org_member, mut_member, cr):
    # Deep copy the original to preserve other attributes
    new_member = copy.deepcopy(org_member)
    D = new_member.args_num

    # Choose a random starting index
    start = np.random.randint(low=0, high=D)

    # Always copy at least one dimension
    L = 0
    i = start

    # Continue copying from mutant while random <= cr and haven't covered all parameters
    while True:
        new_member.chromosomes[i].real_value = mut_member.chromosomes[i].real_value
        L += 1
        i = (i + 1) % D

        # stop if we've copied all dimensions
        if L >= D:
            break

        # draw new rand and test against cr
        if np.random.rand() > cr:
            break

    # For any dimension not yet copied, retain from original
    # We know that only L consecutive positions have been replaced
    # Fill the rest
    for j in range(D):
        # if j not in the copied block
        # compute offset from start in circular sense
        offset = (j - start) % D
        if offset >= L:
            new_member.chromosomes[j].real_value = org_member.chromosomes[j].real_value

    return new_member


def crossing(origin_population: Population, mutated_population: Population, cr, crossing_type: CrossingType):
    if origin_population.size != mutated_population.size:
        print("Populations have different sizes")
        return None

    new_members = []
    for i in range(origin_population.size):
        if crossing_type == CrossingType.BINOMIAL:
            new_member = binomial_crossing_ind(origin_population.members[i], mutated_population.members[i], cr)
        if crossing_type == CrossingType.EXPOTENTIAL:
            new_member = exponential_crossing_ind(origin_population.members[i], mutated_population.members[i], cr)
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
