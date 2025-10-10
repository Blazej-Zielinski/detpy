from detpy.DETAlgs.population_reduction.population_size_reduction_strategy import PopulationSizeReductionStrategy


class LambdaPopulationSizeReduction(PopulationSizeReductionStrategy):
    def __init__(self, reduction_function):
        """
        Initialize the strategy with a custom reduction function.

        Parameters:
        - reduction_function (callable): A function that takes `current_nfe`, `total_nfe`,
          `start_pop_size`, and `min_pop_size` as arguments and returns the new population size.
        """
        if not callable(reduction_function):
            raise ValueError("The reduction_function must be callable.")
        self.reduction_function = reduction_function

    def get_new_population_size(
            self,
            current_nfe: int,
            total_nfe: int,
            start_pop_size: int,
            min_pop_size: int
    ) -> int:
        # Use the provided lambda or callable to calculate the new population size
        return self.reduction_function(current_nfe, total_nfe, start_pop_size, min_pop_size)