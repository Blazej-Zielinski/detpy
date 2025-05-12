from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import IDEData
from detpy.DETAlgs.methods.methods_de import selection, mutation
from detpy.DETAlgs.methods.methods_ide import ide_get_f, ide_get_cr, ide_binomial_crossing
from detpy.models.enums.boundary_constrain import fix_boundary_constraints


class IDE(BaseAlg):
    """
        IDE

        Links:
        https://www.scirp.org/journal/paperinformation?paperid=96749

        References:
        Ma, J. and Li, H. (2019) Research on Rosenbrock Function Optimization Problem Based on Improved Differential
        Evolution Algorithm.
        Journal of Computer and Communications, 7, 107-120. doi: 10.4236/jcc.2019.711008.
    """

    def __init__(self, params: IDEData, db_conn=None, db_auto_write=False):
        super().__init__(IDE.__name__, params, db_conn, db_auto_write)
        self.base_vector_schema = params.base_vector_schema
        self.y = params.y

    def next_epoch(self):
        # Calculate F and CR
        f = ide_get_f(self._epoch_number, self.num_of_epochs)
        cr_arr = ide_get_cr(self._pop)

        # New population after mutation
        v_pop = mutation(self._pop, base_vector_schema=self.base_vector_schema,
                         optimization_type=self.optimization_type,y = self.y, f=f)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = ide_binomial_crossing(self._pop, v_pop, cr_arr)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
