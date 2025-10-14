from detpy.models.member import Member
from detpy.models.stop_condition.stop_condition import StopCondition


class LambdaStopCondition(StopCondition):
    """
    Implementation of StopCondition that accepts a callable
    to define the stopping condition.
    """

    def __init__(self, condition: callable):
        """
        Initialize with a callable that defines the stopping condition.

        :param condition: A callable that takes (nfe, epoch, best_chromosome) as arguments
                          and returns a boolean indicating whether to stop.
        """
        self.condition = condition

    def should_stop(self, nfe: int, epoch: int, best_member: Member) -> bool:
        """
        Evaluate the stopping condition using the provided callable.

        :param nfe: Current number of function evaluations
        :param epoch: Current epoch number
        :param best_member: The best individual in the current population
        :return: True to stop, False to continue
        """
        return self.condition(nfe, epoch, best_member)


"""
Example usage:
if __name__ == "__main__":
    # Define a stopping condition that stops after 1000 epochs.
    stop_condition = LambdaStopCondition(lambda nfe, epoch, best: epoch >= 1000)
"""