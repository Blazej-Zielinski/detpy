from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import OppBasedData
from detpy.DETAlgs.methods.methods_opposition_based import opp_based_generation_jumping
from detpy.DETAlgs.methods.methods_de import mutation, selection, crossing
from detpy.DETAlgs.validator.params_validator import ParameterValidator
from detpy.models.enums.boundary_constrain import fix_boundary_constraints


class OppBasedDE(BaseAlg):
    """
        OppBasedDE

        Links:
        https://ieeexplore.ieee.org/document/4358759

        References:
        S. Rahnamayan, H. R. Tizhoosh and M. M. A. Salama, "Opposition-Based Differential Evolution,"
        in IEEE Transactions on Evolutionary Computation, vol. 12, no. 1, pp. 64-79, Feb. 2008,
        doi: 10.1109/TEVC.2007.894200.
    """

    def __init__(self, params: OppBasedData, db_conn=None, db_auto_write=False, db_writing_interval=5000, verbose=False):
        super().__init__(OppBasedDE.__name__, params, db_conn, db_auto_write, db_writing_interval, verbose)

        self.mutation_factor = params.mutation_factor  # F
        self.crossover_rate = params.crossover_rate  # Cr
        self.crossing_type = params.crossing_type
        self.y = params.y
        self.base_vector_schema = params.base_vector_schema
        self.nfc = 0  # number of function calls
        self.jumping_rate = params.jumping_rate

        ParameterValidator.float_between(
            params.mutation_factor,
            0.0,
            2.0,
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

        ParameterValidator.float_between(
            params.jumping_rate,
            0.0,
            1.0,
            "Jumping rate"
        )

        ParameterValidator.int_times_leq(
            params.y,
            2,
            params.population_size,
            "Y"
        )

    def next_epoch(self):
        # New population after mutation
        v_pop = mutation(self._pop, base_vector_schema=self.base_vector_schema,
                         optimization_type=self.optimization_type, y=self.y, f=self.mutation_factor)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = crossing(self._pop, v_pop, cr=self.crossover_rate, crossing_type=self.crossing_type)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval)
        self.nfc += self.population_size

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Generation jumping
        if opp_based_generation_jumping(new_pop, self.jumping_rate, self._function.eval):
            self.nfc += self.population_size

        # Override data
        self._pop = new_pop
