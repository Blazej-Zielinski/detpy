import numpy as np
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import NMDEData
from detpy.DETAlgs.methods.methods_nmde import nmde_mutation, nmde_selection, nmde_calculate_fm_crm, \
    nmde_binomial_crossing, nmde_update_f_cr
from detpy.models.enums.boundary_constrain import fix_boundary_constraints


class NMDE(BaseAlg):
    """
        NMDE

        Links:
        https://www.sciencedirect.com/science/article/pii/S0898122111000460

        References:
        Dexuan Zou, Haikuan Liu, Liqun Gao, and Steven Li. 2011. A novel modified differential evolution algorithm for
        constrained optimization problems. Comput. Math. Appl. 61, 6 (March, 2011), 1608–1623.
        https://doi.org/10.1016/j.camwa.2011.01.029
    """

    def __init__(self, params: NMDEData, db_conn=None, db_auto_write=False):
        super().__init__(NMDE.__name__, params, db_conn, db_auto_write)

        self.delta_f = params.delta_f
        self.delta_cr = params.delta_cr
        self.sp = params.sp
        self._flags = np.zeros(self.population_size)
        self._f_arr = np.random.uniform(size=self.population_size, low=0, high=2)
        self._cr_arr = np.random.uniform(size=self.population_size, low=0, high=1)
        self._f_set = set()
        self._cr_set = set()

    def next_epoch(self):
        delta_f, delta_cr, sp, flags, f_arr, cr_arr, f_set, cr_set = (
            self.delta_f, self.delta_cr, self.sp, self._flags,
            self._f_arr, self._cr_arr, self._f_set, self._cr_set
        )

        # New population after mutation
        v_pop = nmde_mutation(self._pop, f_arr)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = nmde_binomial_crossing(self._pop, v_pop, cr_arr)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop, better_members_indexes = nmde_selection(self._pop, u_pop)

        for i in range(self.population_size):
            if i in better_members_indexes:
                f_set.add(f_arr[i])
                cr_set.add(cr_arr[i])
            else:
                flags[i] += 1

        f_m, cr_m = nmde_calculate_fm_crm(f_set, cr_set)

        for i in range(self.population_size):
            if flags[i] == sp:
                if f_set != set() and cr_set != set():
                    f_arr[i], cr_arr[i] = nmde_update_f_cr(f_m, cr_m, delta_f, delta_cr)
                else:
                    f_arr[i] = np.random.uniform(low=0, high=2)
                    cr_arr[i] = np.random.uniform(low=0, high=1)
                flags[i] = 0

        # Override data
        self._pop = new_pop
        self.delta_f = delta_f
        self.delta_cr = delta_cr
        self.sp = sp
        self._flags = flags
        self._f_arr = f_arr
        self._cr_arr = cr_arr
        self._f_set = set()
        self._cr_set = set()
