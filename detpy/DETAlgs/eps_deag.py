import copy
import functools
import random

import numpy as np

from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import EPSDEAGData
from detpy.DETAlgs.methods.methods_de import  crossing
from detpy.DETAlgs.methods.methods_eps_de import calculate_epsilon_constrained
from detpy.DETAlgs.methods.methods_eps_deag import mutation, selection, calculate_init_epsilon_level, \
    epsilon_constrained_comparator, gradient_mutation, calculate_epsilon_level
from detpy.models.enums.boundary_constrain import fix_boundary_constraints
from detpy.models.enums.crossingtype import CrossingType
from detpy.models.population import Population

class EPSDEAG(BaseAlg):

    """
          EPSDEAG - Epsilon Constrained Differential Evolution with an Archive and Gradient-Based Mutation

          Links:
          https://ieeexplore.ieee.org/document/5586484

          References:
          Tetsuyuki Takahama; Setsuko Sakai;
          "Constrained optimization by the ε constrained differential evolution with an archive and gradient-based mutation",
          2010 IEEE Congress on Evolutionary Computation,
          18-23 July 2010, Barcelona, Spain doi: 10.1109/CEC.2010.5586484.
    """

    def __init__(self, params: EPSDEAGData, db_conn=None, db_auto_write=False):
        super().__init__(EPSDEAG.__name__, params, db_conn, db_auto_write)
        self.number_of_repeating_de_operations = params.number_of_repeating_de_operations
        self.gradient_mutation_interval = params.gradient_mutation_interval
        self.derivative_method = params.derivative_method
        self.init_mutation_factor = params.init_mutation_factor  # F
        self.init_crossover_rate = params.init_crossover_rate  # Cr
        self.g_funcs = params.g_funcs #Inequality constraints functions
        self.h_funcs = params.h_funcs #Equality constraints functions
        self.theta = params.theta
        self.penalty_power = params.penalty_power
        self.archive_size = params.archive_size
        self.init_epsilon_level = None
        self.archive_members = None
        self.control_generations = params.control_generations
        self.gradient_base_mutation_rate = params.gradient_base_mutation_rate
        self.number_of_repeating_mutation = params.number_of_repeating_mutation
        self.min_epsilon_scaling_factory = 3
        self._initialize_population()
        self.epsilon_constrained = calculate_epsilon_constrained(self._pop, self.g_funcs, self.h_funcs, self.penalty_power)
        self.epsilon_level = self.init_epsilon_level
        calculate_epsilon_scaling_factory = (-5 - np.log(self.init_epsilon_level)) / np.log(0.05)
        self.epsilon_scaling_factor = calculate_epsilon_scaling_factory \
            if calculate_epsilon_scaling_factory >= self.min_epsilon_scaling_factory else self.min_epsilon_scaling_factory

    def _initialize_population(self):
        population = Population(
            lb=self.lb,
            ub=self.ub,
            arg_num=self.nr_of_args,
            size=self.archive_size,
            optimization=self.optimization_type
        )
        population.generate_population()
        population.update_fitness_values(self._function.eval, self.parallel_processing)
        epsilon_constrained = calculate_epsilon_constrained(population, self.g_funcs, self.h_funcs, self.penalty_power)
        self.init_epsilon_level = calculate_init_epsilon_level(epsilon_constrained, self.theta)
        archive_members = sorted(population.members, key=functools.cmp_to_key(
            functools.partial(
                epsilon_constrained_comparator,
                g_funcs=self.g_funcs, h_funcs=self.h_funcs,
                penalty_power=self.penalty_power,
                epsilon_level=self.init_epsilon_level,
                optimization=self.optimization_type)))
        self._pop.members = copy.deepcopy(np.array(archive_members[:self.population_size]))
        self.archive_members = archive_members[self.population_size:]

    def next_epoch(self):
        mutation_factory = self.init_mutation_factor
        crossover_rate = self.init_crossover_rate

        if self._epoch_number > (self.control_generations * 0.95) and self._epoch_number < self.control_generations:
            mutation_factory = 0.3 * self.init_crossover_rate + 0.7
            self.epsilon_scaling_factor = 0.3 * self.epsilon_scaling_factor + 0.7 * self.min_epsilon_scaling_factory

        if random.uniform(0, 1) < 0.05:
            mutation_factory = min(1 + abs(np.random.normal(0, 0.05)), 1.1)
            crossover_rate = random.uniform(0, 1)


        new_pop = self._pop
        selected_child = set()
        gradient_mutation_flag = self._epoch_number % self.gradient_mutation_interval == 0
        for i in range(self.number_of_repeating_de_operations):
            # New population after mutation

            v_pop = mutation(new_pop, self.archive_members, mutation_factory)

            # Apply boundary constrains on population in place
            fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

            # New population after crossing
            u_pop = crossing(new_pop, v_pop, cr=crossover_rate, crossing_type=CrossingType.EXPOTENTIAL)

            gradient_mutation_pop = gradient_mutation(u_pop,  self.number_of_repeating_mutation, self.gradient_base_mutation_rate,
                                                      self.derivative_method, self.g_funcs, self.h_funcs, self.penalty_power, gradient_mutation_flag)

            # Update values before selection
            gradient_mutation_pop.update_fitness_values(self._function.eval, self.parallel_processing)

            u_gradient_epsilon_constrained = calculate_epsilon_constrained(gradient_mutation_pop, self.g_funcs, self.h_funcs, self.penalty_power)

            # Select new population
            new_pop = selection(new_pop, gradient_mutation_pop, self.epsilon_constrained, u_gradient_epsilon_constrained, self.archive_members, selected_child,  self.epsilon_level)

            if len(selected_child) == 0:
                break

        self.epsilon_level = calculate_epsilon_level(self.init_epsilon_level, self._epoch_number, self.control_generations, self.epsilon_scaling_factor)
        self.epsilon_constrained = calculate_epsilon_constrained(new_pop, self.g_funcs, self.h_funcs, self.penalty_power)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1