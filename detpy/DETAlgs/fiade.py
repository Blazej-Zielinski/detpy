from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.methods.methods_fiade import mutation, selection, adapt_parameters, fiade_crossing
from detpy.models.enums.boundary_constrain import fix_boundary_constraints
from detpy.DETAlgs.data.alg_data import FiADEData


class FiADE(BaseAlg):
    """
        FiADE

        Links:
        https://www.sciencedirect.com/science/article/abs/pii/S0020025511001381

        References:
        Arnob Ghosh , Swagatam Das , Aritra Chowdhury , Ritwik Giri (2011)
        An Improved Differential Evolution Algorithm with Fitness-Based Adaptation of the Control Parameters.
        Volume 181, Issue 18, 15 September 2011, Pages 3749-3765
        doi: 10.1016/j.ins.2011.03.010
    """

    def __init__(self, params: FiADEData, db_conn=None, db_auto_write=False):
        super().__init__(FiADE.__name__, params, db_conn, db_auto_write)

        # Initialize mutation factors and crossover rates for the population
        self.mutation_factors = [[params.mutation_factor, False] for _ in range(params.population_size)]
        self.crossover_rates = [[params.crossover_rate, False] for _ in range(params.population_size)]

    def next_epoch(self):
        # Mutation
        v_pop = mutation(self._pop, self.mutation_factors)

        # Apply boundary constraints
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # Crossing
        u_pop = fiade_crossing(self._pop, v_pop, self.crossover_rates)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Selection
        new_pop = selection(self._pop, u_pop, self.mutation_factors, self.crossover_rates)

        # Adapt parameters based on fitness
        adapt_parameters(new_pop, self.mutation_factors, self.crossover_rates)

        # Override population with the newly selected population
        self._pop = new_pop
        self._epoch_number += 1
