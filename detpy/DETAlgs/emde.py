from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import EMDEData
from detpy.DETAlgs.methods.methods_de import selection, crossing
from detpy.DETAlgs.methods.methods_emde import em_mutation
from detpy.models.enums.boundary_constrain import fix_boundary_constraints


class EMDE(BaseAlg):
    """
        EMDE

        Links:
        https://link.springer.com/article/10.1007/s13042-015-0479-6

        References:
        Mohamed, A.W. An efficient modified differential evolution algorithm for solving constrained non-linear
        integer and mixed-integer global optimization problems.
        Int. J. Mach. Learn. & Cyber. 8, 989–1007 (2017). https://doi.org/10.1007/s13042-015-0479-6
    """

    def __init__(self, params: EMDEData, db_conn=None, db_auto_write=False):
        super().__init__(EMDE.__name__, params, db_conn, db_auto_write)

        self.crossover_rate = params.crossover_rate  # Cr
        self.crossing_type = params.crossing_type

    def next_epoch(self):
        # Calculate not constant cr depend on generation number
        v_pop = em_mutation(self._pop)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = crossing(self._pop, v_pop, self.crossover_rate, crossing_type=self.crossing_type)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Override data
        self._pop = new_pop

