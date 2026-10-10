import csv
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from detpy.monitoring.monitor import Monitor, EpochData


class CSVMonitor(Monitor):
    """
    Monitors optimization progress by writing data to two CSV files:
      - general_metrics.csv
      - specific_metrics.csv

    Parameters:
        log_dir: Base output directory.
        experiment_name: Name of the experiment.
        every_nfe: Minimum number of additional function evaluations
                   between logged data points.
        flush_every: Number of logged points between disk synchronizations;
                     1 means synchronizing after every point.
    """

    def __init__(
        self,
        log_dir: str = "runs",
        experiment_name: str = "experiment",
        every_nfe: int = 100,
        flush_every: int = 1,
    ):
        super().__init__()

        if every_nfe < 1:
            raise ValueError("every_nfe must be >= 1")

        if flush_every < 1:
            raise ValueError("flush_every must be >= 1")

        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.general_path = self.log_dir / "general_metrics.csv"
        self.specific_path = self.log_dir / "specific_metrics.csv"

        self.every_nfe = every_nfe
        self.flush_every = flush_every

        self._last_logged_nfe: Optional[int] = None
        self._last_logged_epoch: Optional[int] = None
        self._rows_since_flush = 0
        self._closed = False

        self._general_rows: list[dict[str, Any]] = []
        self._specific_rows: list[dict[str, Any]] = []

        self._general_fields: list[str] = []
        self._specific_fields: list[str] = []

        # Create the files immediately so they are available
        # before the first monitoring point is recorded.
        self._initialize_file(self.general_path)
        self._initialize_file(self.specific_path)

        logging.info(
            "CSVMonitor is writing data to: %s",
            self.log_dir.resolve(),
        )

    @staticmethod
    def _initialize_file(path: Path) -> None:
        """Create an empty CSV file if it does not already exist."""
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as file:
                file.flush()
                os.fsync(file.fileno())

    @staticmethod
    def _ordered_fields(
        existing: list[str],
        row: dict[str, Any],
    ) -> list[str]:
        """Return the existing field names extended with any new keys."""
        fields = list(existing)

        for key in row:
            if key not in fields:
                fields.append(key)

        return fields

    def _sync_file(self, path: Path) -> None:
        """Force the file contents to be synchronized with the disk."""
        with path.open("a", encoding="utf-8") as file:
            file.flush()
            os.fsync(file.fileno())

    def _write_csv(
        self,
        path: Path,
        row: dict[str, Any],
        rows: list[dict[str, Any]],
        fields: list[str],
    ) -> list[str]:
        """
        Write a record to the CSV file immediately.

        If new metrics appear, extend the header and rewrite the file
        while preserving all previously recorded rows.
        """
        new_fields = self._ordered_fields(fields, row)

        if new_fields != fields:
            # New columns require the CSV header to be rewritten.
            rows.append(dict(row))

            with path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=new_fields,
                    extrasaction="ignore",
                )
                writer.writeheader()
                writer.writerows(rows)

                file.flush()
                os.fsync(file.fileno())

            return new_fields

        needs_header = (
            not path.exists()
            or path.stat().st_size == 0
        )

        with path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=new_fields,
                extrasaction="ignore",
            )

            if needs_header:
                writer.writeheader()

            writer.writerow(row)
            file.flush()

            # flush_every=1 means synchronizing after every logged point.
            # For larger values, synchronization occurs after the
            # configured number of writes.
            self._rows_since_flush += 1

            if self._rows_since_flush >= self.flush_every:
                os.fsync(file.fileno())
                self._rows_since_flush = 0

        rows.append(dict(row))
        return new_fields

    def _flush(self) -> None:
        """Perform an additional synchronization of both CSV files."""
        self._sync_file(self.general_path)
        self._sync_file(self.specific_path)
        self._rows_since_flush = 0

    def log_epoch(
        self,
        data: EpochData,
        metrics: Optional[dict[str, Any]] = None,
        force: bool = False,
    ) -> None:
        """
        Log a monitoring point if any of the following conditions apply:
          - This is the first monitoring point.
          - At least every_nfe additional function evaluations have
            occurred since the previous logged point.
          - force=True.

        Duplicate points with the same epoch and NFE values are skipped.
        """
        if self._closed:
            return

        if isinstance(data, EpochData):
            general_row = asdict(data)
        elif isinstance(data, dict):
            general_row = dict(data)
        else:
            raise TypeError(
                "data must be an EpochData instance or a dictionary"
            )

        metrics = metrics or {}

        epoch = int(general_row.get("epoch", 0))
        nfe = int(general_row.get("nfe", 0))

        specific_row = {
            "epoch": epoch,
            "nfe": nfe,
            **metrics,
        }

        # Always log the first monitoring point.
        if self._last_logged_nfe is None:
            should_log = True
        else:
            should_log = (
                nfe - self._last_logged_nfe >= self.every_nfe
            )

        if force:
            should_log = True

        # Do not log the exact same point more than once.
        if (
            self._last_logged_epoch == epoch
            and self._last_logged_nfe == nfe
        ):
            return

        if not should_log:
            return

        self._general_fields = self._write_csv(
            self.general_path,
            general_row,
            self._general_rows,
            self._general_fields,
        )

        self._specific_fields = self._write_csv(
            self.specific_path,
            specific_row,
            self._specific_rows,
            self._specific_fields,
        )

        self._last_logged_epoch = epoch
        self._last_logged_nfe = nfe

        logging.debug(
            "CSVMonitor: logged epoch=%s, nfe=%s",
            epoch,
            nfe,
        )

    def log_final(
        self,
        monitor_data: EpochData,
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Force the final monitoring point to be logged."""
        self.log_epoch(
            monitor_data,
            metrics=metrics,
            force=True,
        )

    def close(self) -> None:
        """Synchronize both files and close the monitor."""
        if self._closed:
            return

        try:
            self._sync_file(self.general_path)
            self._sync_file(self.specific_path)
        except OSError:
            logging.exception(
                "Failed to synchronize CSV files"
            )
        finally:
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False