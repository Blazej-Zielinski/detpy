from detpy.DETAlgs.population_reduction.population_size_reduction_strategy import PopulationSizeReductionStrategy


class FixedPopulationSizeStrategy(PopulationSizeReductionStrategy):
    def get_new_population_size(
            self,
            current_nfe: int,
            total_nfe: int,
            start_pop_size: int,
            min_pop_size: int
    ) -> int:
        return start_pop_size
