from detpy.models.member import Member
from detpy.models.stop_condition.stop_condition import StopCondition


class NeverStopCondition(StopCondition):
    """
    A StopCondition implementation that never stops the algorithm.
    Always returns False, allowing the algorithm to run until its natural limits are reached.
    """

    def should_stop(self, nfe: int, epoch: int, best_member: Member) -> bool:
        """
        Always returns False, indicating the algorithm should not stop running.
        """
        return False
