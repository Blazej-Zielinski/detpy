
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import EPSDEwDCData
from detpy.DETAlgs.methods.methods_de import mutation, crossing
from detpy.DETAlgs.methods.methods_eps_de import selection, calculate_epsilon_constrained
from detpy.DETAlgs.methods.methods_eps_de_w_dc import calculate_t_prime, epsilon_dynamic_control
from detpy.models.enums.basevectorschema import BaseVectorSchema
from detpy.models.enums.boundary_constrain import fix_boundary_constraints
from detpy.models.enums.crossingtype import CrossingType


class EPSDEwDC(BaseAlg):

    """
          EPSDEwDC - Epsilon Constrained Differential Evolution with Dynamic ε-Level Control

          Links:
          https://link.springer.com/chapter/10.1007/978-3-540-68830-3_5

          References:
          Tetsuyuki Takahama, Setsuko Sakai (2008)
          "Constrained Optimization by ε Constrained Differential Evolution with Dynamic ε-Level Control",
          In: Chakraborty, U.K. (eds) Advances in Differential Evolution. Studies in Computational Intelligence,
          vol 143. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-68830-3_5
    """

    def __init__(self, params: EPSDEwDCData, db_conn=None, db_auto_write=False):
        super().__init__(EPSDEwDC.__name__, params, db_conn, db_auto_write)
        self.mutation_factor = params.mutation_factor  # F
        self.crossover_rate = params.crossover_rate  # Cr
        self.g_funcs = params.g_funcs #Inequality constraints functions
        self.h_funcs = params.h_funcs #Equality constraints functions
        self.tolerance_h = params.tolerance_h
        self.penalty_power = params.penalty_power
        self.theta = params.theta if params.theta is not None else int(0.2 * self.population_size)
        self.eta = params.eta
        self.control_generations = params.control_generations
        self.epsilon_constrained = calculate_epsilon_constrained(self._pop, self.g_funcs, self.h_funcs, self.penalty_power, self.tolerance_h)
        self.t_prime = calculate_t_prime(self._epoch_number, self.epsilon_constrained, self.eta, self.control_generations, self.penalty_power)
        self.initial_epsilon_level = epsilon_dynamic_control(self._epoch_number, self.theta, self.epsilon_constrained, self.t_prime, self.control_generations)
        self.epsilon_level = self.initial_epsilon_level


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

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        u_pop_epsilon_constrained = calculate_epsilon_constrained(u_pop, self.g_funcs, self.h_funcs, self.penalty_power, self.tolerance_h)

        # Select new population
        new_pop = selection(self._pop, u_pop, self.epsilon_constrained, u_pop_epsilon_constrained, self.epsilon_level)

        # Override data
        self._pop = new_pop

        self.epsilon_constrained = calculate_epsilon_constrained(self._pop, self.g_funcs, self.h_funcs, self.penalty_power, self.tolerance_h)

        self.t_prime = calculate_t_prime(self._epoch_number, self.epsilon_constrained, self.eta,
                                              self.control_generations, self.penalty_power, self.t_prime,
                                              self.epsilon_level, self.initial_epsilon_level)
        self.epsilon_level = epsilon_dynamic_control(self._epoch_number, self.theta, self.epsilon_constrained,
                                                         self.t_prime, self.control_generations, self.initial_epsilon_level)