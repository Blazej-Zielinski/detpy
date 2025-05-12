import random
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import EIDEData
from detpy.DETAlgs.methods.methods_de import mutation, selection, crossing
from detpy.DETAlgs.methods.methods_eide import eide_adopt_parameters
from detpy.models.enums.boundary_constrain import fix_boundary_constraints


class EIDE(BaseAlg):
    """
        EIDE

        Links:
        https://ieeexplore.ieee.org/document/6390324

        References:
        Z. Dexuan and G. Liqun, "An efficient improved differential evolution algorithm,"
        Proceedings of the 31st Chinese Control Conference, Hefei, China, 2012, pp. 2385-2390.
    """

    def __init__(self, params: EIDEData, db_conn=None, db_auto_write=False):
        super().__init__(EIDE.__name__, params, db_conn, db_auto_write)

        self.mutation_factor = random.uniform(0, 0.6)
        self.crossover_rate = params.crossover_rate_min
        self.crossing_type = params.crossing_type
        self.y = params.y
        self.base_vector_schema = params.base_vector_schema
        self.crossover_rate_min = params.crossover_rate_min
        self.crossover_rate_max = params.crossover_rate_max
        self.generation = None

    def next_epoch(self):
        # New population after mutation
        v_pop = mutation(self._pop, base_vector_schema=self.base_vector_schema,
                         optimization_type=self.optimization_type, y=self.y, f=self.mutation_factor)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = crossing(self._pop, v_pop, cr=self.crossover_rate, crossing_type=self.crossing_type)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        self.mutation_factor, self.crossover_rate = eide_adopt_parameters(self.crossover_rate_min,
                                                                          self.crossover_rate_max, self._epoch_number,
                                                                          self.num_of_epochs)

        # Override data
        self._pop = new_pop

        self._epoch_number += 1
