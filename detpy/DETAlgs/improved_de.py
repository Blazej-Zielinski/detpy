import math
from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import ImprovedDEData
from detpy.DETAlgs.validator.params_validator import ParameterValidator
from detpy.models.enums.boundary_constrain import fix_boundary_constraints
from detpy.DETAlgs.methods.methods_improved_de import mutation, selection, improve_de_binomial_crossing
from detpy.models.enums.optimization import OptimizationType


class ImprovedDE(BaseAlg):
    """
        ImprovedDE

        Links:
        https://link.springer.com/article/10.1007/s00500-023-09080-1

        References:
        Yifeng Lin · Yuer Yang · Yinyan Zhang
        Improved differential evolution with dynamic mutation parameters
        Soft Computing Optimization Published: 17 August 2023 Volume 27, pages 17923–17941, (2023)
        https://doi.org/10.1007/s00500-023-09080-1

    """

    def __init__(self, params: ImprovedDEData, db_conn=None, db_auto_write=False, db_writing_interval=5000, verbose=False, monitor=None):
        super().__init__(ImprovedDE.__name__, params, db_conn, db_auto_write, db_writing_interval, verbose, monitor)

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

        self.initial_mutation_factor = params.mutation_factor  # Initial F value
        self.crossover_rate = params.crossover_rate  # Cr
        self.iteration = 0
        self._successful_mutations_last_epoch = 0
        self._current_f = None

    def get_monitor_metrics(self) -> dict:
        """Return ImprovedDE metrics that change during a run."""
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

    def dynamic_mutation_factor(self, iteration):
        """
        Implements dynamic mutation factor based on Scheme 6:
        FS(k) = 1 - 1 / (1 + exp(-iteration))
        """
        return 1 - 1 / (1 + math.exp(-iteration))

    def next_epoch(self):
        dynamic_fs = self.dynamic_mutation_factor(self.iteration)
        self._current_f = dynamic_fs

        # New population after mutation using dynamic mutation factor
        v_pop = mutation(self._pop, fs=dynamic_fs)

        # Apply boundary constraints on population in place
        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        # New population after crossing
        u_pop = improve_de_binomial_crossing(self._pop, v_pop, cr=self.crossover_rate)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        # Select new population
        new_pop = selection(self._pop, u_pop)
        self._successful_mutations_last_epoch = self._count_successes(new_pop)

        # Override population with the newly selected one
        self._pop = new_pop

        # Increment iteration count
        self.iteration += 1
