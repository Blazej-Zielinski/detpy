from abc import ABC, abstractmethod


class PopulationSizeReductionStrategy(ABC):
    """Interface for population size reduction strategies."""

    @abstractmethod
    def get_new_population_size(self, current_nfe: int, total_nfe: int, start_pop_size: int,
                                min_pop_size: int) -> int:
        """
        Implement this method to define a strategy for reducing the population size.

        Parameters:
        - current_nfe (int): The current nfe number.
        - total_nfe (int): The total number of nfe.
        - start_pop_size (int): The initial population size.
        - min_pop_size (int): The minimum population size.

        Returns:
        - int: The new population size after applying the reduction strategy.

        """
        pass
