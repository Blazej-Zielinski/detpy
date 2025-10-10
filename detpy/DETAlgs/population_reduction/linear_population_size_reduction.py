from detpy.DETAlgs.population_reduction.population_size_reduction_strategy import PopulationSizeReductionStrategy


class LinearPopulationSizeReduction(PopulationSizeReductionStrategy):

    def get_new_population_size(
            self,
            current_nfe: int,
            total_nfe: int,
            start_pop_size: int,
            min_pop_size: int
    ) -> int:
        new_size = int(
            round(start_pop_size - (current_nfe / total_nfe) * (start_pop_size - min_pop_size))
        )
        return new_size
