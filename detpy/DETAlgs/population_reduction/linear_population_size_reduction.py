from detpy.DETAlgs.population_reduction.population_size_reduction_strategy import PopulationSizeReductionStrategy


class LinearPopulationSizeReduction(PopulationSizeReductionStrategy):
    def get_new_population_size(
            self,
            current_epoch: int,
            total_epochs: int,
            start_pop_size: int,
            min_pop_size: int
    ) -> int:
        # Difference between the initial and minimum population sizes
        delta = start_pop_size - min_pop_size

        # Compute the total number of evaluations
        # This is required since we work with the number of epochs, not evaluations
        # and original LSHADE and related SHADE algorithms define the reduction strategy
        # based on the number of evaluations
        total_evaluations = sum(
            int(start_pop_size - (gen / total_epochs) * delta)
            for gen in range(total_epochs)
        )

        # Compute the number of evaluations completed up to the current epoch
        current_eval = sum(
            int(start_pop_size - (gen / total_epochs) * delta)
            for gen in range(current_epoch)
        )

        # Calculate the new population size following the original LSHADE
        # and related SHADE algorithm definitions
        new_size = int(start_pop_size - (current_eval / total_evaluations) * delta)
        return new_size
