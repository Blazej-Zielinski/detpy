import random
import numpy as np
import copy

from detpy.models.population import Population
from detpy.models.enums.optimization import OptimizationType
from detpy.models.enums.mutation import mutation_rand_1


def delb_mutation(population: Population):
    members = population.members.tolist()
    drew_members = [random.sample(members, 3) for _ in range(population.size)]
    drew_fs = [random.uniform(-1, -0.4) if random.random() < 0.5 else random.uniform(0.4, 1)
               for _ in range(population.size)]

    new_members = [mutation_rand_1(selected_members, f)
                   for selected_members, f in zip(drew_members, drew_fs)]

    new_population = Population(
        lb=population.lb,
        ub=population.ub,
        arg_num=population.arg_num,
        size=population.size,
        optimization=population.optimization
    )
    new_population.members = np.array(new_members)

    return new_population


def delb_selection(origin_population: Population, modified_population: Population, w: float, fitness_func):
    if origin_population.size != modified_population.size:
        print("Selection: populations have different sizes")
        return None

    if origin_population.optimization != modified_population.optimization:
        print("Selection: populations have different optimization types")
        return None

    best_member = origin_population.get_best_members(1)[0]
    optimization = origin_population.optimization

    new_members = []
    for i in range(origin_population.size):
        if optimization == OptimizationType.MINIMIZATION:
            if origin_population.members[i] <= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
            elif random.uniform(0, 1) < w and modified_population.members[i] > best_member:
                r = best_member - (modified_population.members[i] - best_member)
                r.calculate_fitness_fun(fitness_fun=fitness_func)
                if r < modified_population.members[i]:
                    new_members.append(copy.deepcopy(r))
                else:
                    c = best_member + (modified_population.members[i] - best_member) * 0.5
                    c.calculate_fitness_fun(fitness_fun=fitness_func)
                    if c < modified_population.members[i]:
                        new_members.append(copy.deepcopy(c))
                    else:
                        new_members.append(copy.deepcopy(modified_population.members[i]))
            else:
                new_members.append(copy.deepcopy(modified_population.members[i]))

        elif optimization == OptimizationType.MAXIMIZATION:
            if origin_population.members[i] >= modified_population.members[i]:
                new_members.append(copy.deepcopy(origin_population.members[i]))
            elif random.uniform(0, 1) < w and modified_population.members[i] < best_member:
                r = best_member - (modified_population.members[i] - best_member)
                r.calculate_fitness_fun(fitness_fun=fitness_func)
                if r > modified_population.members[i]:
                    new_members.append(copy.deepcopy(r))
                else:
                    c = best_member + (modified_population.members[i] - best_member) * 0.5
                    c.calculate_fitness_fun(fitness_fun=fitness_func)
                    if c > modified_population.members[i]:
                        new_members.append(copy.deepcopy(c))
                    else:
                        new_members.append(copy.deepcopy(modified_population.members[i]))
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
