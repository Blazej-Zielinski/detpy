import copy
import random

import numpy as np

from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import  EPSADEData
from detpy.DETAlgs.methods.methods_de import mutation, crossing
from detpy.DETAlgs.methods.methods_eps_ade import control_epsilon_level, adaptive_de_operation
from detpy.DETAlgs.methods.methods_eps_de import calculate_epsilon_constrained, epsilon_level_comparisons, \
    epsilon_constrained_method
from detpy.DETAlgs.methods.methods_eps_deag import calculate_init_epsilon_level, calculate_epsilon_level
from detpy.models.enums.basevectorschema import BaseVectorSchema
from detpy.models.enums.boundary_constrain import fix_boundary_constraints
from detpy.models.enums.crossingtype import CrossingType
from detpy.models.population import Population


class EPSADE(BaseAlg):

    """
          EPSADE - Constrained Adaptive Differential Evolution

          Links:
          https://ieeexplore.ieee.org/abstract/document/5586545

          References:
          Tetsuyuki Takahama; Setsuko Sakai;
          "Efficient constrained optimization by the ε constrained adaptive differential evolution",
          2010 IEEE Congress on Evolutionary Computation,
          18-23 July 2010, Barcelona, Spain doi: 10.1109/CEC.2010.5586545.
    """

    def __init__(self, params: EPSADEData, db_conn=None, db_auto_write=False):
        super().__init__(EPSADE.__name__, params, db_conn, db_auto_write)
        self.init_mutation_factor = params.init_mutation_factor  # F
        self.init_crossover_rate = params.init_crossover_rate  # Cr
        self.mu_mutation_factory = self.init_mutation_factor
        self.mu_crossover_rate = self.init_crossover_rate
        self.number_of_successful_operation = 0
        self.mutation_factor_sum = 0
        self.crossover_rate_sum = 0
        self.mutation_factor_perturbation_width = params.mutation_factor_perturbation_width
        self.crossover_rate_perturbation_width = params.crossover_rate_perturbation_width
        self.weight_of_update = np.clip(params.weight_of_update, 1e-12, 1)
        self.truncation_mechanism_factory = params.truncation_mechanism_factory #ap
        self.g_funcs = params.g_funcs #Inequality constraints functions
        self.h_funcs = params.h_funcs #Equality constraints functions
        self.control_generations = params.control_generations
        self.epsilon_scaling_factor = params.epsilon_scaling_factor
        self.penalty_power = params.penalty_power
        self.theta = params.theta
        self.epsilon_constrained = calculate_epsilon_constrained(self._pop, self.g_funcs, self.h_funcs,
                                                                 self.penalty_power)
        self.init_epsilon_level = calculate_init_epsilon_level(self.epsilon_constrained, self.theta)
        self.epsilon_level = self.init_epsilon_level
        self.penalty_power = params.penalty_power

    def selection(self, modified_population: Population) -> Population | None:

        if self._pop.size != modified_population.size:
            print("Selection: populations have different sizes")
            return None

        if self._pop.optimization != modified_population.optimization:
            print("Selection: populations have different optimization types")
            return None

        optimization = self._pop.optimization
        modified_epsilon_constrained = calculate_epsilon_constrained(modified_population, self.g_funcs, self.h_funcs, self.penalty_power)
        new_members = []
        for i in range(self._pop.size):
            if epsilon_level_comparisons(modified_population.members[i], self._pop.members[i],
                                         modified_epsilon_constrained[i], self.epsilon_constrained[i], self.epsilon_level,
                                         optimization):
                new_members.append(copy.deepcopy(modified_population.members[i]))
            else:
                mutation_factor = np.clip(
                    self.mu_mutation_factory + self.mutation_factor_perturbation_width * random.uniform(-0.5, 0.5)
                    , 0.4, 0.9)
                crossover_rate = np.clip(
                    self.mu_crossover_rate + self.crossover_rate_perturbation_width * random.uniform(-0.5, 0.5)
                    , 0, 1)
                new_member = adaptive_de_operation(self._pop, self._pop.members[i], mutation_factor, crossover_rate, self._function)
                new_member_epsilon_constrained = epsilon_constrained_method(new_member.get_chromosomes(), self.g_funcs, self.h_funcs, self.penalty_power)

                if epsilon_level_comparisons(new_member, self._pop.members[i],
                                         new_member_epsilon_constrained, self.epsilon_constrained[i], self.epsilon_level,
                                         optimization):
                    new_members.append(new_member)
                    self.number_of_successful_operation += 1
                    self.mutation_factor_sum += mutation_factor
                    self.crossover_rate_sum += crossover_rate
                else:
                    new_members.append(copy.deepcopy(self._pop.members[i]))

        new_population = Population(
            lb=self._pop.lb,
            ub=self._pop.ub,
            arg_num=self._pop.arg_num,
            size=self._pop.size,
            optimization=self._pop.optimization
        )
        new_population.members = np.array(new_members)
        return new_population


    def update_adaptive_parameters(self):
        if self.number_of_successful_operation > 0:
            self.mu_mutation_factory = (1 - self.weight_of_update) * self.mu_mutation_factory + (self.weight_of_update * self.mutation_factor_sum) / self.number_of_successful_operation
            self.mu_crossover_rate = (1 - self.weight_of_update) * self.mu_crossover_rate + (self.weight_of_update * self.crossover_rate_sum) / self.number_of_successful_operation

    def next_epoch(self):
        self.number_of_successful_operation = 0
        self.mutation_factor_sum = 0
        self.crossover_rate_sum = 0

        # New population after mutation
        v_pop = mutation(self._pop, base_vector_schema=BaseVectorSchema.RAND,
                         optimization_type=self.optimization_type,
                         y=1,
                         f=self.init_mutation_factor)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = crossing(self._pop, v_pop, cr=self.init_crossover_rate, crossing_type=CrossingType.EXPOTENTIAL)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = self.selection(u_pop)

        self.update_adaptive_parameters()

        self.epsilon_level = calculate_epsilon_level(self.init_epsilon_level, self.nfe,
                                                     self.control_generations, self.epsilon_scaling_factor)
        self.epsilon_constrained = calculate_epsilon_constrained(new_pop, self.g_funcs, self.h_funcs,
                                                                 self.penalty_power)

        self.epsilon_level = control_epsilon_level(self.epsilon_level, self.epsilon_constrained, self.truncation_mechanism_factory, self.population_size)

        # Override data
        self._pop = new_pop
