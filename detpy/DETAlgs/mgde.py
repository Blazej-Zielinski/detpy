from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import MGDEData
from detpy.DETAlgs.methods.methods_de import selection, crossing
from detpy.DETAlgs.methods.methods_mgde import mgde_mutation, mgde_adapt_threshold
from detpy.models.enums.boundary_constrain import fix_boundary_constraints


class MGDE(BaseAlg):
    """
        MGDE

        Links:
        https://link.springer.com/article/10.1007/s10479-022-04641-3

        References:
        Zouache, D., Ben Abdelaziz, F. MGDE: a many-objective guided differential evolution with strengthened dominance
        relation and bi-goal evolution. Ann Oper Res (2022). https://doi.org/10.1007/s10479-022-04641-3
    """

    def __init__(self, params: MGDEData, db_conn=None, db_auto_write=False):
        super().__init__(MGDE.__name__, params, db_conn, db_auto_write)

        self.mutation_factor_f = params.mutation_factor_f
        self.mutation_factor_k = params.mutation_factor_k
        self.crossover_rate = params.crossover_rate
        self.crossing_type = params.crossing_type
        self.threshold = params.threshold
        self.mu = params.mu
        self.generation = 1

    def next_epoch(self):
        # New population after mutation
        v_pop = mgde_mutation(self._pop, self.generation, self.num_of_epochs, self.mutation_factor_f,
                              self.mutation_factor_k)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = crossing(self._pop, v_pop, cr=self.crossover_rate, crossing_type=self.crossing_type)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        mgde_adapt_threshold(new_pop, self.threshold, self.mu, self._function.eval)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
