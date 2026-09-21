from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import FDDEData
from detpy.DETAlgs.methods.methods_fdde import (
    calculate_fitness_ranking,
    calculate_diversity_ranking,
    calculate_final_ranking,
    fdde_mutation,
)
from detpy.DETAlgs.methods.methods_de import crossing, selection
from detpy.DETAlgs.validator.params_validator import ParameterValidator
from detpy.models.enums.boundary_constrain import fix_boundary_constraints
from detpy.models.enums.ranking_type import RankingType


class FDDE(BaseAlg):
    """
        FDDE

        Links:
        https://www.sciencedirect.com/science/article/abs/pii/S2210650220304697

        References:
        J. Cheng, Z. Pan, H. Liang, Z. Gao, J. Gao,
        Differential evolution algorithm with fitness and diversity ranking-based mutation operator,
        Swarm and Evolutionary Computation, Volume 61, 2021, 100816
    """

    def __init__(self, params: FDDEData, db_conn=None, db_auto_write=False, db_writing_interval=5000, verbose=False):
        super().__init__(FDDE.__name__, params, db_conn, db_auto_write, db_writing_interval, verbose)

        ParameterValidator.float_between(
            params.mutation_factor,
            0.0,
            1.0,
            "mutation_factor"
        )

        ParameterValidator.float_between(
            params.crossover_rate,
            0.0,
            1.0,
            "crossover_rate"
        )

        self.mutation_factor = params.mutation_factor
        self.crossover_rate = params.crossover_rate
        self.crossing_type = params.crossing_type
        self.ranking_type = params.ranking_type

    def next_epoch(self):
        if self.ranking_type == RankingType.FITNESS_ONLY:
            final_rankings = calculate_fitness_ranking(self._pop)
        elif self.ranking_type == RankingType.DIVERSITY_ONLY:
            final_rankings = calculate_diversity_ranking(self._pop)
        else:
            fr = calculate_fitness_ranking(self._pop)
            dr = calculate_diversity_ranking(self._pop)
            w = 0.2 + 0.6 * min(self.nfe / self.nfe_max, 1.0)
            final_rankings = calculate_final_ranking(fr, dr, w)

        v_pop = fdde_mutation(self._pop, final_rankings, self.mutation_factor)

        fix_boundary_constraints(v_pop, self.boundary_constraints_fun)

        u_pop = crossing(self._pop, v_pop, cr=self.crossover_rate, crossing_type=self.crossing_type)

        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)

        new_pop = selection(self._pop, u_pop)

        self._pop = new_pop
