import csv
import numbers
from pathlib import Path
from typing import Dict, Optional

from detpy.monitoring.monitor import Monitor, EpochData


class CSVMonitor(Monitor):
    """
    CSV monitor for DETPy optimization algorithms.

    Standard metrics are stored as regular CSV columns.

    Algorithm-specific metrics are stored under:

        algorithm_metrics/<metric_name>
    """

    BASE_FIELDS = [
        "epoch",
        "nfe",
        "best_fitness",
        "mean_fitness",
        "std_fitness",
        "population_min_fitness",
        "population_max_fitness",
        "population_diversity",
        "epoch_time",
        "evaluations_per_second",
    ]

    def __init__(
        self,
        log_dir="runs",
        experiment_name=None,
        filename="metrics.csv",
        every_epochs=None,
        every_nfe=None,
        flush_every=1,
    ):
        if every_epochs is not None and every_epochs < 1:
            raise ValueError(
                "every_epochs must be >= 1"
            )

        if every_nfe is not None and every_nfe < 1:
            raise ValueError(
                "every_nfe must be >= 1"
            )

        if flush_every < 1:
            raise ValueError(
                "flush_every must be >= 1"
            )

        if (
            every_epochs is None
            and every_nfe is None
        ):
            every_epochs = 1

        self.every_epochs = every_epochs
        self.every_nfe = every_nfe
        self.flush_every = flush_every

        self._next_nfe = (
            every_nfe
            if every_nfe is not None
            else None
        )

        self._logged_since_flush = 0

        self.log_dir = Path(log_dir)

        if experiment_name:
            self.log_dir = (
                self.log_dir / experiment_name
            )

        self.log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.filepath = (
            self.log_dir / filename
        )

        self.file = self.filepath.open(
            mode="w",
            newline="",
            encoding="utf-8",
        )

        self.writer = None
        self._fieldnames = None

        # Rows received before the CSV schema
        # can be determined.
        self._pending_rows = []

    def _should_log(
        self,
        data: EpochData,
    ) -> bool:
        """
        Determine whether the current epoch
        should be logged.
        """

        if (
            self.every_epochs is not None
            and data.epoch % self.every_epochs == 0
        ):
            return True

        if (
            self._next_nfe is not None
            and data.nfe >= self._next_nfe
        ):
            return True

        return False

    def _update_nfe_threshold(
        self,
        data: EpochData,
    ):
        """
        Move the NFE threshold forward.
        """

        if self._next_nfe is None:
            return

        while data.nfe >= self._next_nfe:
            self._next_nfe += self.every_nfe

    def _build_row(
        self,
        data: EpochData,
        metrics: Optional[Dict[str, float]],
    ) -> dict:
        """
        Build one CSV row.
        """

        row = {
            "epoch": data.epoch,
            "nfe": data.nfe,
            "best_fitness": data.best_fitness,
            "mean_fitness": data.mean_fitness,
            "std_fitness": data.std_fitness,
            "population_min_fitness": (
                data.population_min_fitness
            ),
            "population_max_fitness": (
                data.population_max_fitness
            ),
            "population_diversity": (
                data.population_diversity
            ),
            "epoch_time": data.epoch_time,
            "evaluations_per_second": (
                data.evaluations_per_second
            ),
        }

        if metrics:
            for name, value in metrics.items():
                if value is None:
                    continue

                if not isinstance(
                    value,
                    numbers.Real,
                ):
                    continue

                row[
                    f"algorithm_metrics/{name}"
                ] = float(value)

        return row

    def _initialize_writer(
        self,
        rows: list,
    ):
        """
        Initialize CSV writer from buffered rows.
        """

        fieldnames = []

        for row in rows:
            for field in row:
                if field not in fieldnames:
                    fieldnames.append(field)

        self._fieldnames = fieldnames

        self.writer = csv.DictWriter(
            self.file,
            fieldnames=self._fieldnames,
            extrasaction="ignore",
        )

        self.writer.writeheader()

        for row in rows:
            self.writer.writerow(row)

        self.file.flush()

        self._logged_since_flush = len(rows)

    def log_epoch(
        self,
        data: EpochData,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Log one optimization epoch to CSV.
        """

        if not self._should_log(data):
            return

        row = self._build_row(
            data=data,
            metrics=metrics,
        )

        # -------------------------------------------------
        # CSV schema is not known yet.
        #
        # Keep initial rows in memory until we receive
        # algorithm-specific metrics.
        # -------------------------------------------------

        if self.writer is None:

            self._pending_rows.append(row)

            has_algorithm_metrics = any(
                key.startswith(
                    "algorithm_metrics/"
                )
                for key in row
            )

            if has_algorithm_metrics:
                self._initialize_writer(
                    self._pending_rows
                )

                self._pending_rows.clear()

            self._update_nfe_threshold(data)

            return

        # -------------------------------------------------
        # Writer already initialized.
        # -------------------------------------------------

        new_fields = [
            field
            for field in row
            if field not in self._fieldnames
        ]

        if new_fields:
            raise ValueError(
                "CSVMonitor received new metric "
                "fields after the CSV header was "
                "created: "
                f"{new_fields}"
            )

        self.writer.writerow(row)

        self._logged_since_flush += 1

        self._update_nfe_threshold(data)

        if (
            self._logged_since_flush
            >= self.flush_every
        ):
            self.file.flush()
            self._logged_since_flush = 0

    def close(self):
        """
        Flush and close the CSV file.

        If no algorithm-specific metrics were
        ever received, pending rows are still
        written using only the standard schema.
        """

        if self.file.closed:
            return

        # No algorithm-specific metrics were ever
        # received. Write pending standard rows.
        if (
            self.writer is None
            and self._pending_rows
        ):
            self._initialize_writer(
                self._pending_rows
            )

            self._pending_rows.clear()

        self.file.flush()
        self.file.close()