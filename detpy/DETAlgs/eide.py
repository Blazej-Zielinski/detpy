import random
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import EIDEData
from detpy.DETAlgs.methods.methods_de import mutation, selection, crossing
from detpy.DETAlgs.methods.methods_eide import eide_adopt_parameters
from detpy.DETAlgs.validator.params_validator import ParameterValidator
from detpy.models.enums.optimization import OptimizationType
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

    def __init__(self, params: EIDEData, db_conn=None, db_auto_write=False, db_writing_interval=5000, verbose=False, monitor=None):
        super().__init__(EIDE.__name__, params, db_conn, db_auto_write, db_writing_interval, verbose, monitor)

        ParameterValidator.float_between(
            params.crossover_rate_min, 0.0, 1.0, "Crossover rate min"
        )

        ParameterValidator.float_between(
            params.crossover_rate_max, 0.0, 1.0, "Crossover rate max"
        )

        ParameterValidator.min_max(
            params.crossover_rate_min,
            params.crossover_rate_max,
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

        self.mutation_factor = random.uniform(0, 0.6)
        self.crossover_rate = params.crossover_rate_min
        self.crossing_type = params.crossing_type
        self.y = params.y
        self.base_vector_schema = params.base_vector_schema
        self.crossover_rate_min = params.crossover_rate_min
        self.crossover_rate_max = params.crossover_rate_max
        self.generation = None

        self._successful_mutations_last_epoch = 0
        self._current_f = None

    def get_monitor_metrics(self) -> dict:
        """Return EIDE metrics that change during a run."""
        return {
            "successful_mutations": self._successful_mutations_last_epoch,
            "successful_mutations_rate": (
                self._successful_mutations_last_epoch / self.population_size
                if self.population_size > 0
                else 0.0
            ),
            "f_mean": 0.0 if self._current_f is None else float(self._current_f),
        }

    def _count_successes(self, new_population) -> int:
        count = 0
        minimize = self._pop.optimization == OptimizationType.MINIMIZATION
        for old_member, new_member in zip(self._pop.members, new_population.members):
            if (new_member < old_member) if minimize else (new_member > old_member):
                count += 1
        return count

    def next_epoch(self):
        self._current_f = self.mutation_factor
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
        self._successful_mutations_last_epoch = self._count_successes(new_pop)

        self.mutation_factor, self.crossover_rate = eide_adopt_parameters(self.crossover_rate_min,
                                                                          self.crossover_rate_max, self.nfe,
                                                                          self.nfe_max)

        # Override data
        self._pop = new_pop
