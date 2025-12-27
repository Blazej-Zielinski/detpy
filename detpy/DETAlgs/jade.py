from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import JADEData
from detpy.DETAlgs.methods.methods_jade import jade_adapt_mutation_factors, jade_binomial_crossing, \
    jade_adapt_crossover_rates, jade_mutation, jade_selection, jade_reduce_archive, draw_norm_dist_within_bounds, \
    draw_cauchy_dist_within_bounds
from detpy.models.enums.boundary_constrain import fix_boundary_constraints


class JADE(BaseAlg):
    """
        JADE

        Links:
        https://ieeexplore.ieee.org/document/5208221

        References:
        J. Zhang and A. C. Sanderson, "JADE: Adaptive Differential Evolution With Optional External Archive,"
        in IEEE Transactions on Evolutionary Computation, vol. 13, no. 5, pp. 945-958, Oct. 2009,
        doi: 10.1109/TEVC.2009.2014613.
    """

    def __init__(self, params: JADEData, db_conn=None, db_auto_write=False):
        super().__init__(JADE.__name__, params, db_conn, db_auto_write)

        self.archive_size = params.archive_size
        self.archive = []

        self.mutation_factor_mean = params.mutation_factor_mean
        self.mutation_factor_std = params.mutation_factor_std
        self.mutation_factors = draw_cauchy_dist_within_bounds(self.mutation_factor_mean, self.mutation_factor_std,
                                                               self.population_size)
        self.success_mutation_factors = []

        self.crossover_rate_mean = params.crossover_rate_mean
        self.crossover_rate_std = params.crossover_rate_std
        self.crossover_rate_low = params.crossover_rate_low
        self.crossover_rate_high = params.crossover_rate_high
        self.crossover_rates = draw_norm_dist_within_bounds(self.crossover_rate_mean, self.crossover_rate_std,
                                                            self.crossover_rate_low, self.crossover_rate_high,
                                                            self.population_size)
        self.success_crossover_rates = []

        self.c = params.c
        self.p = params.p

    def next_epoch(self):
        # New population after mutation
        v_pop = jade_mutation(self._pop, self.mutation_factors, self.p, self.archive)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = jade_binomial_crossing(self._pop, v_pop, self.crossover_rates)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = jade_selection(self._pop, u_pop, self.mutation_factors, self.crossover_rates,
                                 self.success_mutation_factors, self.success_crossover_rates, self.archive)

        jade_reduce_archive(self.archive, self.archive_size)

        self.mutation_factors, self.mutation_factor_mean = jade_adapt_mutation_factors(self.c,
                                                                                       self.mutation_factor_mean,
                                                                                       self.mutation_factor_std,
                                                                                       self.population_size,
                                                                                       self.success_mutation_factors)

        self.crossover_rates, self.crossover_rate_mean = jade_adapt_crossover_rates(self.c, self.crossover_rate_mean,
                                                                                    self.crossover_rate_std,
                                                                                    self.crossover_rate_low,
                                                                                    self.crossover_rate_high,
                                                                                    self.population_size,
                                                                                    self.success_crossover_rates)

        # Override data
        self._pop = new_pop
