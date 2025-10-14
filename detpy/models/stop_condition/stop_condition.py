from abc import ABC, abstractmethod
from detpy.models.member import Member


class StopCondition(ABC):
    """
    Interface for controlling the execution flow of an algorithm.
    It allows defining custom stopping criteria for optimization or simulation processes.
    """

    @abstractmethod
    def should_stop(self, nfe: int, epoch: int, best_member: Member) -> bool:
        """
        Called after each epoch (or iteration) to determine whether the algorithm should stop.

        :param nfe: Current number of function evaluations (Number of Function Evaluations)
        :param epoch: Current epoch number
        :param best_member: The best individual (member) in the current population
        :return: True to stop the algorithm, False to continue it
        """
        pass