
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import EPSDEData
from detpy.DETAlgs.methods.methods_de import mutation, crossing
from detpy.DETAlgs.methods.methods_eps_de import selection, calculate_epsilon_constrained
from detpy.models.enums.basevectorschema import BaseVectorSchema
from detpy.models.enums.boundary_constrain import fix_boundary_constraints
from detpy.models.enums.crossingtype import CrossingType


class EPSDE(BaseAlg):

    """
          EPSDE - Epsilon Constrained Differential Evolution

          Links:
          https://ieeexplore.ieee.org/abstract/document/4274215

          References:
          Tetsuyuki Takahama; Setsuko Sakai; Noriyuki Iwane
          "Solving Nonlinear Constrained Optimization Problems by the Epsilon Constrained Differential Evolution",
          2006 IEEE International Conference on Systems, Man and Cybernetics,
          08-11 October 2006, Taipei,Taiwan doi: 10.1109/ICSMC.2006.385209.
    """

    def __init__(self, params: EPSDEData, db_conn=None, db_auto_write=False):
        super().__init__(EPSDE.__name__, params, db_conn, db_auto_write)
        self.mutation_factor = params.mutation_factor  # F
        self.crossover_rate = params.crossover_rate  # Cr
        self.g_funcs = params.g_funcs #Inequality constraints functions
        self.h_funcs = params.h_funcs #Equality constraints functions
        self.epsilon_level = params.epsilon_level
        self.penalty_power = params.penalty_power


    def next_epoch(self):
        pop_epsilon_constrained = calculate_epsilon_constrained(self._pop, self.g_funcs, self.h_funcs, self.penalty_power)

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

        u_pop_epsilon_constrained = calculate_epsilon_constrained(u_pop, self.g_funcs, self.h_funcs, self.penalty_power)

        # Select new population
        new_pop = selection(self._pop, u_pop, pop_epsilon_constrained, u_pop_epsilon_constrained, self.epsilon_level)

        # Override data
        self._pop = new_pop
