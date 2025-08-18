
import numpy as np

from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import EPSDEGData
from detpy.DETAlgs.methods.methods_de import crossing, mutation
from detpy.DETAlgs.methods.methods_eps_de import calculate_epsilon_constrained, selection
from detpy.DETAlgs.methods.methods_eps_deag import calculate_init_epsilon_level, calculate_epsilon_level
from detpy.DETAlgs.methods.methods_eps_deg import gradient_mutation
from detpy.models.enums.basevectorschema import BaseVectorSchema
from detpy.models.enums.boundary_constrain import fix_boundary_constraints, BoundaryFixing
from detpy.models.enums.crossingtype import CrossingType

class EPSDEG(BaseAlg):

    """
          EPSDEG - Epsilon Constrained Differential Evolution with Gradient-Based Mutation

          Links:
          https://link.springer.com/chapter/10.1007/978-3-642-00619-7_3

          References:
          T. Takahama and S. Sakai,
          “Solving difficult constrained optimization problems by the ε constrained differential evolution with gradient based mutation”
           in Constraint-Handling in Evolutionary Optimization,
           E. Mezura-Montes, Ed. Springer-Verlag, 2009, pp. 51–72.
    """

    def __init__(self, params: EPSDEGData, db_conn=None, db_auto_write=False):
        super().__init__(EPSDEG.__name__, params, db_conn, db_auto_write)
        self.derivative_method = params.derivative_method
        self.mutation_factor = params.mutation_factor  # F
        self.crossover_rate = params.crossover_rate  # Cr
        self.g_funcs = params.g_funcs #Inequality constraints functions
        self.h_funcs = params.h_funcs #Equality constraints functions
        self.gradient_mutation_interval = params.gradient_mutation_interval
        self.theta = params.theta
        self.penalty_power = params.penalty_power
        self.epsilon_constrained = calculate_epsilon_constrained(self._pop, self.g_funcs, self.h_funcs,
                                                                 self.penalty_power)
        self.init_epsilon_level = calculate_init_epsilon_level(self.epsilon_constrained, self.theta)
        self.control_generations = params.control_generations
        self.gradient_base_mutation_rate = params.gradient_base_mutation_rate
        self.min_epsilon_scaling_factory = 3
        self.epsilon_level = self.init_epsilon_level
        calculate_epsilon_scaling_factory = (-5 - np.log(self.init_epsilon_level)) / np.log(0.05)
        self.epsilon_scaling_factor = calculate_epsilon_scaling_factory \
            if calculate_epsilon_scaling_factory >= self.min_epsilon_scaling_factory else self.min_epsilon_scaling_factory


    def next_epoch(self):
        # New population after mutation
        v_pop = mutation(self._pop, base_vector_schema=BaseVectorSchema.RAND,
                         optimization_type=self.optimization_type,
                         y=1,
                         f=self.mutation_factor)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = crossing(self._pop, v_pop, cr=self.crossover_rate, crossing_type=CrossingType.EXPOTENTIAL)

        fix_boundary_constraints(v_pop, BoundaryFixing.REFLECTION_BACK)

        gradient_mutation_pop = gradient_mutation(u_pop, self.gradient_mutation_interval, self.gradient_base_mutation_rate,
                                                  self.epsilon_level, self.derivative_method, self.g_funcs, self.h_funcs, self.penalty_power)

        # Update values before selection
        gradient_mutation_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        u_gradient_epsilon_constrained = calculate_epsilon_constrained(gradient_mutation_pop, self.g_funcs, self.h_funcs, self.penalty_power)

        # Select new population
        new_pop = selection(self._pop, gradient_mutation_pop, self.epsilon_constrained, u_gradient_epsilon_constrained, self.epsilon_level)

        self.epsilon_level = calculate_epsilon_level(self.init_epsilon_level, self._epoch_number, self.control_generations, self.epsilon_scaling_factor)
        self.epsilon_constrained = calculate_epsilon_constrained(new_pop, self.g_funcs, self.h_funcs, self.penalty_power)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1