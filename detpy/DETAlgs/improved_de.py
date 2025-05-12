import math
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import ImprovedDEData
from detpy.models.enums.boundary_constrain import fix_boundary_constraints
from detpy.DETAlgs.methods.methods_improved_de import mutation, selection, improve_de_binomial_crossing


class ImprovedDE(BaseAlg):
    """
        ImprovedDE

        Links:
        https://link.springer.com/article/10.1007/s00500-023-09080-1

        References:
        Yifeng Lin · Yuer Yang · Yinyan Zhang
        Improved differential evolution with dynamic mutation parameters
        Soft Computing Optimization Published: 17 August 2023 Volume 27, pages 17923–17941, (2023)
        https://doi.org/10.1007/s00500-023-09080-1

    """

    def __init__(self, params: ImprovedDEData, db_conn=None, db_auto_write=False):
        super().__init__(ImprovedDE.__name__, params, db_conn, db_auto_write)
        self.initial_mutation_factor = params.mutation_factor  # Initial F value
        self.crossover_rate = params.crossover_rate  # Cr
        self.iteration = 0

    def dynamic_mutation_factor(self, iteration):
        """
        Implements dynamic mutation factor based on Scheme 6:
        FS(k) = 1 - 1 / (1 + exp(-iteration))
        """
        return 1 - 1 / (1 + math.exp(-iteration))

    def next_epoch(self):
        dynamic_fs = self.dynamic_mutation_factor(self.iteration)

        # New population after mutation using dynamic mutation factor
        v_pop = mutation(self._pop, fs=dynamic_fs)

        # Apply boundary constraints on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = improve_de_binomial_crossing(self._pop, v_pop, cr=self.crossover_rate)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Override population with the newly selected one
        self._pop = new_pop

        # Increment iteration count
        self.iteration += 1
        self._epoch_number += 1
