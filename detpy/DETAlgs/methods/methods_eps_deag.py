import copy
import random

import numpy as np
import scipy as sp
from autograd import jacobian
import autograd.numpy as anp

from detpy.DETAlgs.methods.methods_de import mutation_ind
from detpy.DETAlgs.methods.methods_eps_de import epsilon_level_comparisons, epsilon_constrained_method
from detpy.models.enums.derivative_method import DerivativeMethod
from detpy.models.enums.optimization import OptimizationType
from detpy.models.member import Member
from detpy.models.population import Population


def mutation(population: Population, archive: np.ndarray, f: float):
    new_members = []
    for i in range(population.size):
        selected_members = random.sample(population.members.tolist(), 3)
        if random.uniform(0, 1) > 0.05:
            selected_members[2] = random.choice(np.union1d(population.members, archive))
        new_member = mutation_ind(selected_members[0], selected_members[1:], f)
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


def selection(origin_population: Population, modified_population: Population,
              origin_epsilon_constrained : list[float], modified_epsilon_constrained : list[float], archive : list[float], selected_child : set,
              epsilon_level : int):
    """
       Perform selection operation for the population.
       Parameters:
       - origin_population (Population): The original population.
       - modified_population (Population): The modified population.
       - origin_epsilon_constrained (list[float]): The original population epsilon constrained population.
       - modified_epsilon_constrained (list[float]): The modified population epsilon constrained population.
       - archive (list[float]): The archive.
       - selected_child (list[int]): The indexes of selected children.
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
        if i in selected_child:
            new_members.append(copy.deepcopy(origin_population.members[i]))
        elif epsilon_level_comparisons(modified_population.members[i], origin_population.members[i],
                                  modified_epsilon_constrained[i], origin_epsilon_constrained[i],
                                  epsilon_level, optimization):
            new_members.append(copy.deepcopy(modified_population.members[i]))
            selected_child.discard(i)
        else:
            selected_child.add(i)
            selected_archive = random.choice(archive)
            archive.remove(selected_archive)
            archive.append(copy.deepcopy(modified_population.members[i]))
            new_members.append(copy.deepcopy(origin_population.members[i]))


    new_population = Population(
        lb=origin_population.lb,
        ub=origin_population.ub,
        arg_num=origin_population.arg_num,
        size=origin_population.size,
        optimization=origin_population.optimization
    )
    new_population.members = np.array(new_members)
    return new_population

def epsilon_constrained_comparator(first_member : Member, second_member : Member, g_funcs : list, h_funcs : list, penalty_power: int, epsilon_level : int, optimization : OptimizationType):
    epsilon_constrained_for_first_member = epsilon_constrained_method(first_member.get_chromosomes(), g_funcs, h_funcs, penalty_power)
    epsilon_constrained_for_second_member = epsilon_constrained_method(second_member.get_chromosomes(), g_funcs, h_funcs, penalty_power)
    if epsilon_level_comparisons(first_member, second_member,
                                 epsilon_constrained_for_first_member, epsilon_constrained_for_second_member,
                                 epsilon_level, optimization):
        return -1
    elif epsilon_level_comparisons(second_member, first_member,
                                   epsilon_constrained_for_second_member, epsilon_constrained_for_first_member,
                                   epsilon_level, optimization):
        return 1
    return 0

def calculate_init_epsilon_level(epsilon_constrained : list, theta : float):
    if theta > 1:
        return theta * np.max(epsilon_constrained)
    else:
        sorted_epsilon_constrained = sorted(epsilon_constrained, reverse=True)
        epsilon_constrained_length = len(sorted_epsilon_constrained)
        last_index = epsilon_constrained_length - 1
        index_theta = min(int(np.floor(theta * last_index)), last_index)
        return sorted_epsilon_constrained[index_theta]

def calculate_epsilon_level(init_epsilon_level : float, epoch_number : int, control_generations : int, epsilon_scaling_factor : int):
    if epoch_number >= control_generations:
        return 0
    else:
        return init_epsilon_level * (1 - (epoch_number / control_generations)) ** epsilon_scaling_factor


def inverse_gradient_constraints(gradient_constraint : np.ndarray):
    if gradient_constraint.shape[0] == gradient_constraint.shape[1] and np.linalg.det(gradient_constraint) != 0:
        return np.linalg.inv(gradient_constraint)

    return np.linalg.pinv(gradient_constraint)

def derivative_numeric(chromosome : list, g_funcs : list, h_funcs : list, eta: float = 1e-5):
    grad = np.zeros_like(chromosome)
    for i in range(len(chromosome)):
        e = np.zeros_like(chromosome)
        e[i] = 1.0
        grad[i] = (constraint_functions((chromosome + eta * e).tolist(), DerivativeMethod.NUMERIC, g_funcs, h_funcs) - constraint_functions(chromosome, DerivativeMethod.NUMERIC, g_funcs, h_funcs)) / eta
    return grad.reshape(1, -1)

def derivative_symbolic(chromosome: list, g_funcs : list, h_funcs : list):
    n = len(chromosome)
    variables = sp.symbols(f'x0:{n}')

    constraints = constraint_functions(variables, DerivativeMethod.SYMBOLIC, g_funcs, h_funcs)

    gradient_list = [[sp.diff(g, var) for var in variables] for g in constraints.tolist()]

    grad_fun = sp.lambdify([variables], gradient_list, "numpy")

    val = grad_fun(chromosome)

    return np.array(val)

def constraint_functions(chromosomes : list, derivative_method : DerivativeMethod, g_funcs : list, h_funcs : list):
    if DerivativeMethod.AUTOMATIC == derivative_method:
        inequality_constraints = anp.array([g(chromosomes) for g in g_funcs])
        equality_constraints = anp.array([h(chromosomes) for h in h_funcs])
        return anp.concatenate((inequality_constraints, equality_constraints))

    inequality_constraints = ([g(chromosomes) for g in g_funcs])
    equality_constraints = ([h(chromosomes) for h in h_funcs])
    return np.concatenate((inequality_constraints, equality_constraints))

def gradient_mutation(pop_population: Population, number_of_repeating_mutation : int, derivative_method : DerivativeMethod, g_funcs : list, h_funcs : list, penalty_power : int, is_gradient_mutation : bool):
    new_members = []
    if is_gradient_mutation:
        for i in range(len(pop_population.members)):
            member = copy.deepcopy(pop_population.members[i])
            if  random.uniform(0, 1) < number_of_repeating_mutation:
                h = 0
                epsilon_constrain = epsilon_constrained_method(member.get_chromosomes(), g_funcs, h_funcs, penalty_power)
                while h < number_of_repeating_mutation and epsilon_constrain > 0:
                    chromosomes = member.get_chromosomes()
                    gradient_constraint = None
                    if DerivativeMethod.NUMERIC == derivative_method:
                        gradient_constraint = derivative_numeric(chromosomes, g_funcs, h_funcs)
                    elif DerivativeMethod.SYMBOLIC == derivative_method:
                        gradient_constraint = derivative_symbolic(chromosomes, g_funcs, h_funcs)
                    else:
                        gradient_constraint = jacobian(constraint_functions)(anp.array(chromosomes))

                    inv_gradient_constraint = inverse_gradient_constraints(gradient_constraint)

                    delta_x =  -np.dot(-inv_gradient_constraint, epsilon_constrain)

                    for j in range(member.args_num):
                        member.chromosomes[j].real_value = member.chromosomes[j].real_value + delta_x.item(j)

                    epsilon_constrain = epsilon_constrained_method(member.get_chromosomes(), g_funcs, h_funcs, penalty_power)
                    h += 1

            new_members.append(member)

        new_population = Population(
            lb=pop_population.lb,
            ub=pop_population.ub,
            arg_num=pop_population.arg_num,
            size=pop_population.size,
            optimization=pop_population.optimization
        )
        new_population.members = np.array(new_members)
        return new_population

    return copy.deepcopy(pop_population)
