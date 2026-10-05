from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import DEData
from detpy.DETAlgs.methods.methods_de import mutation, selection, crossing
from detpy.DETAlgs.validator.params_validator import ParameterValidator
from detpy.models.enums.boundary_constrain import fix_boundary_constraints


class DE(BaseAlg):
    """
        The original version of different evolution

        Links:
        https://link.springer.com/article/10.1023/A:1008202821328

        References:
        Storn, R., Price, K.
        Differential Evolution – A Simple and Efficient Heuristic for global Optimization over Continuous Spaces.
        Journal of Global Optimization 11, 341–359 (1997).
        https://doi.org/10.1023/A:1008202821328

    """

    def __init__(self, params: DEData, db_conn=None, db_auto_write=False, db_writing_interval=5000, verbose=False, monitor=None):
        super().__init__(DE.__name__, params, db_conn, db_auto_write, db_writing_interval, verbose, monitor)

        ParameterValidator.float_between(
            params.mutation_factor,
            0.0,
            1.0,
            "Mutation factor"
        )

        ParameterValidator.float_between(
            params.crossover_rate,
            0.0,
            1.0,
            "Crossover rate"
        )

        ParameterValidator.int_min(
            params.y,
            1,
            "Y"
        )

        ParameterValidator.int_times_leq(
            params.y,
            2,
            params.population_size,
            "Y"
        )

        self.mutation_factor = params.mutation_factor  # F
        self.crossover_rate = params.crossover_rate  # Cr
        self.crossing_type = params.crossing_type
        self.y = params.y
        self.base_vector_schema = params.base_vector_schema

    def next_epoch(self):
        # New population after mutation
        v_pop = mutation(self._pop, base_vector_schema=self.base_vector_schema,
                         optimization_type=self.optimization_type,
                         y=self.y,
                         f=self.mutation_factor)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = crossing(self._pop, v_pop, cr=self.crossover_rate, crossing_type=self.crossing_type)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Override data
        self._pop = new_pop
