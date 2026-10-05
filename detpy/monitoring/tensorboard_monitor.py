from pathlib import Path
import numbers
import time

from detpy.monitoring.monitor import Monitor, EpochData


class TensorBoardMonitor(Monitor):
    """
    TensorBoard monitor for DETPy optimization algorithms.

    Requires only the `tensorboard` package.

    Parameters
    ----------
    log_dir : str or Path
        Directory where TensorBoard logs will be stored.

    experiment_name : str, optional
        Optional experiment subdirectory.

    flush_every : int
        Flush TensorBoard events to disk every N seconds.

    every_epochs : int, optional
        Log every N epochs.

    every_nfe : int, optional
        Log when NFE reaches or exceeds the next NFE threshold.

    Notes
    -----
    If both `every_epochs` and `every_nfe` are specified,
    logging occurs when either condition is satisfied.

    The first real measurement is also duplicated at NFE=0
    as an initial visualization baseline. This does not
    represent an actual objective-function evaluation.

    Standard metrics are stored under the following namespaces:

        fitness/
        population/
        performance/
        algorithm/

    Algorithm-specific metrics are stored under:

        algorithm_metrics/

    This allows different optimization algorithms to expose
    their own metrics without mixing them with the standard
    metrics shared by all algorithms.
    """

    def __init__(
        self,
        log_dir="runs",
        experiment_name=None,
        flush_every=10,
        every_epochs=None,
        every_nfe=None,
    ):
        if every_epochs is not None and every_epochs < 1:
            raise ValueError("every_epochs must be >= 1")

        if every_nfe is not None and every_nfe < 1:
            raise ValueError("every_nfe must be >= 1")

        if flush_every < 1:
            raise ValueError("flush_every must be >= 1")

        # If no logging interval is specified,
        # log every epoch.
        if every_epochs is None and every_nfe is None:
            every_epochs = 1

        try:
            from tensorboard.compat.proto.event_pb2 import Event
            from tensorboard.compat.proto.summary_pb2 import Summary
            from tensorboard.summary.writer.event_file_writer import (
                EventFileWriter
            )
        except ImportError as exc:
            raise ImportError(
                "TensorBoard support requires the 'tensorboard' package. "
                "Install it with: pip install tensorboard"
            ) from exc

        self.Event = Event
        self.Summary = Summary

        self.every_epochs = every_epochs
        self.every_nfe = every_nfe

        # Fixed NFE thresholds:
        # 1000, 2000, 3000, ...
        self._next_nfe = (
            every_nfe
            if every_nfe is not None
            else None
        )

        # Tracks whether the NFE=0 baseline has already
        # been written.
        self._initial_baseline_logged = False

        self.log_dir = Path(log_dir)

        if experiment_name:
            self.log_dir = self.log_dir / experiment_name

        self.log_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.writer = EventFileWriter(
            str(self.log_dir),
            flush_secs=flush_every
        )

    def _should_log(self, data: EpochData) -> bool:
        """
        Determine whether the current epoch should be logged.
        """

        # Epoch-based logging.
        if (
            self.every_epochs is not None
            and data.epoch % self.every_epochs == 0
        ):
            return True

        # NFE-based logging.
        if (
            self._next_nfe is not None
            and data.nfe >= self._next_nfe
        ):
            return True

        return False

    def _update_nfe_threshold(self, data: EpochData):
        """
        Move the NFE threshold forward after logging.

        Example
        -------
        every_nfe=1000

        data.nfe=1096
        next threshold -> 2000
        """

        if self._next_nfe is None:
            return

        while data.nfe >= self._next_nfe:
            self._next_nfe += self.every_nfe

    def _add_scalar(
        self,
        tag: str,
        value: numbers.Real,
        step: int,
    ):
        """
        Add a scalar value to TensorBoard.
        """

        summary = self.Summary(
            value=[
                self.Summary.Value(
                    tag=tag,
                    simple_value=float(value)
                )
            ]
        )

        event = self.Event(
            wall_time=time.time(),
            step=int(step),
            summary=summary
        )

        self.writer.add_event(event)

    def _log_metrics(
        self,
        metrics,
        step: int,
        prefix: str = "algorithm_metrics",
    ):
        """
        Log algorithm-specific scalar metrics.

        Parameters
        ----------
        metrics : dict, optional
            Dictionary containing metric names and scalar values.

        step : int
            TensorBoard step, normally NFE.

        prefix : str
            TensorBoard namespace used for the metrics.

        Notes
        -----
        Metric names may contain `/` to create additional
        TensorBoard groups.

        Example
        -------
        Input:

            {
                "strategy_probability/cb": 0.25,
                "parameters/mean_F": 0.52,
            }

        Results in:

            algorithm_metrics/strategy_probability/cb
            algorithm_metrics/parameters/mean_F
        """

        if not metrics:
            return

        for name, value in metrics.items():

            if value is None:
                continue

            # Supports Python numeric types as well as
            # NumPy scalar types such as np.float64 / np.int64.
            if not isinstance(value, numbers.Real):
                continue

            self._add_scalar(
                tag=f"{prefix}/{name}",
                value=value,
                step=step
            )

    def _log_initial_baseline(self, data: EpochData):
        """
        Log the initial visualization baseline at NFE=0.

        The values are copied from the first real measurement.
        This point does not represent an actual objective-function
        evaluation.

        Algorithm-specific metrics are intentionally not logged
        at NFE=0 because there may be no meaningful algorithm-
        specific state before the first optimization epoch.
        """

        if self._initial_baseline_logged:
            return

        # ---------------------------------------------------------
        # Fitness
        # ---------------------------------------------------------

        self._add_scalar(
            "fitness/best",
            data.best_fitness,
            0
        )

        self._add_scalar(
            "fitness/mean",
            data.mean_fitness,
            0
        )

        self._add_scalar(
            "fitness/std",
            data.std_fitness,
            0
        )

        # ---------------------------------------------------------
        # Population
        # ---------------------------------------------------------

        if data.population_diversity is not None:
            self._add_scalar(
                "population/diversity",
                data.population_diversity,
                0
            )

        if data.population_min_fitness is not None:
            self._add_scalar(
                "population/min_fitness",
                data.population_min_fitness,
                0
            )

        if data.population_max_fitness is not None:
            self._add_scalar(
                "population/max_fitness",
                data.population_max_fitness,
                0
            )

        # ---------------------------------------------------------
        # Performance
        # ---------------------------------------------------------

        self._add_scalar(
            "performance/epoch_time",
            0.0,
            0
        )

        if data.evaluations_per_second is not None:
            self._add_scalar(
                "performance/evaluations_per_second",
                0.0,
                0
            )

        # ---------------------------------------------------------
        # Algorithm
        # ---------------------------------------------------------

        self._add_scalar(
            "algorithm/epoch",
            0,
            0
        )

        self._initial_baseline_logged = True

    def log_epoch(
        self,
        data: EpochData,
        metrics=None,
    ):
        """
        Log one optimization epoch to TensorBoard.

        Parameters
        ----------
        data : EpochData
            Standard metrics shared by all algorithms.

        metrics : dict, optional
            Additional algorithm-specific scalar metrics.

            Example:

                {
                    "strategy_probability/cb": 0.25,
                    "parameters/mean_F": 0.52,
                }

            They will be logged as:

                algorithm_metrics/strategy_probability/cb
                algorithm_metrics/parameters/mean_F
        """

        if not self._should_log(data):
            return

        # The first real data point is used to create
        # a visualization baseline at NFE=0.
        if (
            not self._initial_baseline_logged
            and data.nfe > 0
        ):
            self._log_initial_baseline(data)

        # NFE is used as the global TensorBoard step.
        step = data.nfe

        # ---------------------------------------------------------
        # Fitness
        # ---------------------------------------------------------

        self._add_scalar(
            "fitness/best",
            data.best_fitness,
            step
        )

        self._add_scalar(
            "fitness/mean",
            data.mean_fitness,
            step
        )

        self._add_scalar(
            "fitness/std",
            data.std_fitness,
            step
        )

        # ---------------------------------------------------------
        # Population
        # ---------------------------------------------------------

        if data.population_diversity is not None:
            self._add_scalar(
                "population/diversity",
                data.population_diversity,
                step
            )

        if data.population_min_fitness is not None:
            self._add_scalar(
                "population/min_fitness",
                data.population_min_fitness,
                step
            )

        if data.population_max_fitness is not None:
            self._add_scalar(
                "population/max_fitness",
                data.population_max_fitness,
                step
            )

        # ---------------------------------------------------------
        # Performance
        # ---------------------------------------------------------

        self._add_scalar(
            "performance/epoch_time",
            data.epoch_time,
            step
        )

        if data.evaluations_per_second is not None:
            self._add_scalar(
                "performance/evaluations_per_second",
                data.evaluations_per_second,
                step
            )

        # ---------------------------------------------------------
        # Algorithm
        # ---------------------------------------------------------

        self._add_scalar(
            "algorithm/epoch",
            data.epoch,
            step
        )

        # ---------------------------------------------------------
        # Algorithm-specific metrics
        # ---------------------------------------------------------

        self._log_metrics(
            metrics=metrics,
            step=step,
            prefix="algorithm_metrics"
        )

        # Update the next NFE threshold after logging.
        self._update_nfe_threshold(data)

    def close(self):
        """
        Close the TensorBoard event writer.
        """

        self.writer.close()