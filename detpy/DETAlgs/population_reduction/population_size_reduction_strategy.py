from abc import ABC, abstractmethod


class PopulationSizeReductionStrategy(ABC):
    """Interface for population size reduction strategies."""

    @abstractmethod
    def get_total_number_of_evaluations(self,
                                        total_epochs: int,
                                        start_pop_size: int,
                                        min_pop_size: int) -> int:
        """
        Implement this method to calculate the total number of evaluations based on the reduction strategy.

        Parameters:
        - total_epochs (int): The total number of epochs.
        - start_pop_size (int): The initial population size.
        - min_pop_size (int): The minimum population size.

        Returns:
        - int: The total number of evaluations.
        """
        pass

    @abstractmethod
    def get_new_population_size(self, current_epoch: int, total_epochs: int, start_pop_size: int,
                                min_pop_size: int) -> int:
        """
        Implement this method to define a strategy for reducing the population size.

        Parameters:
        - current_epoch (int): The current epoch number.
        - total_epochs (int): The total number of epochs.
        - start_pop_size (int): The initial population size.
        - min_pop_size (int): The minimum population size.

        Returns:
        - int: The new population size after applying the reduction strategy.

        """
        pass
