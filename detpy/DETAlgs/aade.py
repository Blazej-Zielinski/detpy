import numpy as np

from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import AADEData
from detpy.DETAlgs.methods.methods_aade import aade_mutation, aade_crossing, aade_selection, \
    aade_adapat_parameters
from detpy.DETAlgs.validator.params_validator import ParameterValidator
from detpy.models.enums.boundary_constrain import fix_boundary_constraints


class AADE(BaseAlg):
    """
        AADE

        Links:
        https://ieeexplore.ieee.org/document/8819749

        References:
        V. Sharma, S. Agarwal and P. K. Verma, "Auto Adaptive Differential Evolution Algorithm,"
        2019 3rd International Conference on Computing Methodologies and Communication (ICCMC),
        Erode, India, 2019, pp. 958-963, doi: 10.1109/ICCMC.2019.8819749.
    """

    def __init__(self, params: AADEData, db_conn=None, db_auto_write=False, db_writing_interval=5000, verbose=False, monitor=None):
        super().__init__(AADE.__name__, params, db_conn, db_auto_write, db_writing_interval, verbose, monitor)

        ParameterValidator.positive_float(
            params.mutation_factor,
            "Mutation factor"
        )

        ParameterValidator.positive_float(
            params.crossover_rate,
            "Crossover rate"
        )

        self.mutation_factors = [[params.mutation_factor, False] for _ in range(params.population_size)]
        self.crossover_rates = [[params.crossover_rate, False] for _ in range(params.population_size)]

        self._successful_mutations_last_epoch = 0
        self._current_f = []
        self._current_cr = []

    def get_monitor_metrics(self) -> dict:
        """Return AADE metrics that change during a run."""
        return {
            "successful_mutations": self._successful_mutations_last_epoch,
            "successful_mutations_rate": (
                self._successful_mutations_last_epoch / self.population_size
                if self.population_size > 0
                else 0.0
            ),
            "f_mean": float(np.mean(self._current_f)) if self._current_f else 0.0,
            "f_std": float(np.std(self._current_f)) if self._current_f else 0.0,
            "cr_mean": float(np.mean(self._current_cr)) if self._current_cr else 0.0,
            "cr_std": float(np.std(self._current_cr)) if self._current_cr else 0.0,
        }

    def next_epoch(self):
        self._current_f = [factor[0] for factor in self.mutation_factors]
        self._current_cr = [rate[0] for rate in self.crossover_rates]
        # New population after mutation
        v_pop = aade_mutation(self._pop, mutation_factors=self.mutation_factors)

        # Apply boundary constrains on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = aade_crossing(self._pop, v_pop, crossover_rates=self.crossover_rates)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = aade_selection(self._pop, u_pop, self.mutation_factors, self.crossover_rates)
        self._successful_mutations_last_epoch = sum(
            1 for factor in self.mutation_factors if not factor[1]
        )

        aade_adapat_parameters(self._pop, self.mutation_factors, self.crossover_rates)

        # Override data
        self._pop = new_pop
