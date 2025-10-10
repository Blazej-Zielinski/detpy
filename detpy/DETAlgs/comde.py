from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import COMDEData
from detpy.DETAlgs.methods.methods_comde import calculate_cr, comde_mutation
from detpy.DETAlgs.methods.methods_de import selection, crossing
from detpy.models.enums.boundary_constrain import fix_boundary_constraints


class COMDE(BaseAlg):
    """
        COMDE

        Links:
        https://www.sciencedirect.com/science/article/pii/S0020025512000278

        References:
        Mohamed, A. W., & Sabry, H. Z. (2012). Constrained optimization based on modified differential evolution algorithm.
        In Information Sciences (Vol. 194, pp. 171–208). Elsevier BV. https://doi.org/10.1016/j.ins.2012.01.008

    """

    def __init__(self, params: COMDEData, db_conn=None, db_auto_write=False):
        super().__init__(COMDE.__name__, params, db_conn, db_auto_write)

        self.mutation_factor = params.mutation_factor  # F
        self.crossover_rate = params.crossover_rate  # Cr
        self.crossing_type = params.crossing_type

    def next_epoch(self):
        # Calculate not constant cr depend on generation number
        cr = calculate_cr(self.nfe, self.nfe_max)

        # New population after mutation
        v_pop = comde_mutation(self._pop)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = crossing(self._pop, v_pop, cr=cr, crossing_type=self.crossing_type)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Override data
        self._pop = new_pop
