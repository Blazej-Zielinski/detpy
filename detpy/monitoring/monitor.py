from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class EpochData:
    """
    Standardized data produced after a completed optimization epoch.

    These fields are common to all optimization algorithms.
    Algorithm-specific metrics should be passed separately
    through the `metrics` argument of `Monitor.log_epoch()`.
    """

    algorithm_name: str

    epoch: int
    nfe: int

    best_fitness: float
    mean_fitness: float
    std_fitness: float

    epoch_time: float

    population_diversity: Optional[float] = None

    population_min_fitness: Optional[float] = None
    population_max_fitness: Optional[float] = None

    evaluations_per_second: Optional[float] = None


class Monitor(ABC):
    """
    Base interface for optimization run monitors.

    Concrete implementations can send metrics to:
    - TensorBoard
    - CSV
    - MLflow
    - W&B
    - custom dashboards

    `EpochData` contains metrics shared by all algorithms.

    Algorithm-specific metrics can be supplied through the
    optional `metrics` dictionary.
    """

    @abstractmethod
    def log_epoch(
        self,
        data: EpochData,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Called after every completed epoch.

        Parameters
        ----------
        data : EpochData
            Standard metrics shared by all algorithms.

        metrics : dict, optional
            Additional algorithm-specific scalar metrics.

            Example
            -------
            {
                "strategy_probability/cb": 0.35,
                "strategy_probability/ce": 0.20,
                "parameters/mean_F": 0.61,
            }

            Metric names should use `/` to create logical
            TensorBoard groups.
        """
        pass

    def close(self):
        """
        Release resources.

        Implementations that don't need cleanup can leave this unchanged.
        """
        pass

