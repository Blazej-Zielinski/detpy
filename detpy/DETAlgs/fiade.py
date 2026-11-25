import numpy as np

from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.methods.methods_fiade import mutation, selection, adapt_parameters, fiade_crossing
from detpy.DETAlgs.validator.params_validator import ParameterValidator
from detpy.models.enums.boundary_constrain import fix_boundary_constraints
from detpy.DETAlgs.data.alg_data import FiADEData


class FiADE(BaseAlg):
    """
        FiADE

        Links:
        https://www.sciencedirect.com/science/article/abs/pii/S0020025511001381

        References:
        Arnob Ghosh , Swagatam Das , Aritra Chowdhury , Ritwik Giri (2011)
        An Improved Differential Evolution Algorithm with Fitness-Based Adaptation of the Control Parameters.
        Volume 181, Issue 18, 15 September 2011, Pages 3749-3765
        doi: 10.1016/j.ins.2011.03.010
    """

    def __init__(self, params: FiADEData, db_conn=None, db_auto_write=False, db_writing_interval=5000, verbose=False, monitor=None):
        super().__init__(FiADE.__name__, params, db_conn, db_auto_write, db_writing_interval, verbose, monitor)

        ParameterValidator.positive_float(
            params.mutation_factor,
            "Mutation factor"
        )

        ParameterValidator.float_between(
            params.crossover_rate,
            0.0,
            1.0,
            "Crossover rate"
        )

        # Initialize mutation factors and crossover rates for the population
        self.mutation_factors = [[params.mutation_factor, False] for _ in range(params.population_size)]
        self.crossover_rates = [[params.crossover_rate, False] for _ in range(params.population_size)]

        self._successful_mutations_last_epoch = 0
        self._current_f = []
        self._current_cr = []

    def get_monitor_metrics(self) -> dict:
        """Return FiADE metrics that change during a run."""
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
        # Mutation
        v_pop = mutation(self._pop, self.mutation_factors)

        # Apply boundary constraints
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # Crossing
        u_pop = fiade_crossing(self._pop, v_pop, self.crossover_rates)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Selection
        new_pop = selection(self._pop, u_pop, self.mutation_factors, self.crossover_rates)
        self._successful_mutations_last_epoch = sum(
            1 for factor in self.mutation_factors if not factor[1]
        )

        # Adapt parameters based on fitness
        adapt_parameters(new_pop, self.mutation_factors, self.crossover_rates)

        # Override population with the newly selected population
        self._pop = new_pop
