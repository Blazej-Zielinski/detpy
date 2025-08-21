
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import EPSRDEData
from detpy.DETAlgs.methods.methods_eps_de import selection, calculate_epsilon_constrained
from detpy.DETAlgs.methods.methods_eps_deag import calculate_init_epsilon_level, \
    calculate_epsilon_level
from detpy.DETAlgs.methods.methods_eps_rde import mutation, crossing, create_ranks, calculate_mutation_factors, calculate_crossover_rates
from detpy.models.enums.basevectorschema import BaseVectorSchema
from detpy.models.enums.boundary_constrain import fix_boundary_constraints
from detpy.models.enums.crossingtype import CrossingType


class EPSRDE(BaseAlg):

    """
          EPSRDE -  Epsilon Constrained Rank-Based Differential Evolution

          Links:
          https://ieeexplore.ieee.org/abstract/document/4274215

          References:
          Tetsuyuki Takahama and Setsuko Sakai
          "Efficient Constrained Optimization by the ε Constrained Rank-Based Differential Evolution",
          2012 IEEE Congress on Evolutionary Computation,
          10-15 June 2012, Brisbane, QLD, Australia doi: 10.1109/CEC.2012.6256111.
    """

    def __init__(self, params: EPSRDEData, db_conn=None, db_auto_write=False):
        super().__init__(EPSRDE.__name__, params, db_conn, db_auto_write)
        self.crossing_type = params.crossing_type
        self.min_mutation_factor = params.min_mutation_factor  # min F
        self.max_mutation_factor = params.max_mutation_factor  # max F
        self.min_crossover_rate = params.min_crossover_rate  # min Cr
        self.max_crossover_rate = params.max_crossover_rate  # max Cr
        self.g_funcs = params.g_funcs #Inequality constraints functions
        self.h_funcs = params.h_funcs #Equality constraints functions
        self.control_generations =  params.control_generations
        self.epsilon_scaling_factor = params.epsilon_scaling_factor
        self.penalty_power = params.penalty_power
        self.theta = params.theta
        self.epsilon_constrained = calculate_epsilon_constrained(self._pop, self.g_funcs, self.h_funcs, self.penalty_power)
        self.init_epsilon_level = calculate_init_epsilon_level(self.epsilon_constrained, self.theta)
        self.epsilon_level = self.init_epsilon_level
        self.ranks = create_ranks(self._pop, self.epsilon_constrained)

    def next_epoch(self):
        mutation_factors = calculate_mutation_factors(self._pop, self.ranks, self.min_mutation_factor, self.max_mutation_factor)
        crossover_rates = calculate_crossover_rates(self._pop, self.ranks, self.min_crossover_rate, self.max_crossover_rate)

        # New population after mutation
        v_pop = mutation(self._pop, base_vector_schema=BaseVectorSchema.RAND,
                         optimization_type=self.optimization_type,
                         y=1,
                         f=mutation_factors)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = crossing(self._pop, v_pop, cr=crossover_rates, crossing_type=CrossingType.EXPOTENTIAL)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        u_pop_epsilon_constrained = calculate_epsilon_constrained(u_pop, self.g_funcs, self.h_funcs, self.penalty_power)

        # Select new population
        new_pop = selection(self._pop, u_pop, self.epsilon_constrained, u_pop_epsilon_constrained, self.epsilon_level)

        self.epsilon_level = calculate_epsilon_level(self.init_epsilon_level, self._epoch_number,
                                                     self.control_generations, self.epsilon_scaling_factor)
        self.epsilon_constrained = calculate_epsilon_constrained(new_pop, self.g_funcs, self.h_funcs,
                                                                 self.penalty_power)

        self.ranks = create_ranks(self._pop, self.epsilon_constrained)
        # Override data
        self._pop = new_pop

        self._epoch_number += 1