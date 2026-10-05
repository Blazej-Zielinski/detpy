import copy
import time
import traceback
import uuid
from abc import ABC, abstractmethod
import datetime
from typing import Optional

from colorama import Fore, Style
from tqdm import tqdm
import numpy as np

from detpy.DETAlgs.validator.params_validator import ParameterValidator
from detpy.database.database_connector import SQLiteConnector
from detpy.DETAlgs.data.alg_data import BaseData
from detpy.helpers.database_helper import get_table_name, format_individuals
from detpy.helpers.metric_helper import MetricHelper
from detpy.models.algorithm_result import AlgorithmResult
from detpy.models.fitness_function import FitnessFunctionWrapper, FitnessFunction
from detpy.models.population import Population
from detpy.helpers.logger import Logger
from detpy.monitoring.monitor import EpochData, Monitor


def example_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    return (x1 - 1) ** 2 + (x2 - 2) ** 2 + (x3 - 3) ** 2 + \
        (x4 - 4) ** 2 + (x5 - 5) ** 2 + (x6 - 6) ** 2 + \
        (x7 - 7) ** 2 + (x8 - 8) ** 2 + (x9 - 9) ** 2 + \
        (x10 - 10) ** 2


class BaseAlg(ABC):

    def __init__(
            self,
            name,
            params: BaseData,
            db_conn=None,
            db_auto_write=False,
            db_writing_interval=5000,
            verbose=False,
            monitor: Optional[Monitor] = None
    ):
        self.name = name

        # The NFE as the main stopping condition works with an accuracy
        # equal to the number of chromosomes in the population,
        # because the main iteration of the algorithm operates
        # over the entire population.
        self.nfe_max = params.max_nfe
        self.additional_stop_criteria = params.additional_stop_criteria
        self._epoch_number = 0

        self._origin_pop = None
        self._pop = None
        self._total_init_time = None

        self.population_size = params.population_size
        self.nr_of_args = params.dimension
        self.lb = params.lb
        self.ub = params.ub
        self.optimization_type = params.optimization_type
        self.boundary_constraints_fun = params.boundary_constraints_fun

        if params.function is None:
            base_fitness = FitnessFunction(func=example_function)
            self._function = FitnessFunctionWrapper(base_fitness)
        else:
            wrapped_function = FitnessFunctionWrapper(func=params.function)
            self._function = wrapped_function

        self._database = SQLiteConnector(db_conn) if db_conn is not None else None
        self.db_auto_write = db_auto_write
        self.log_population = params.log_population
        self.parallel_processing = params.parallel_processing
        self.database_table_name = None
        self.show_plots = params.show_plots

        self.db_writing_interval = db_writing_interval

        # Use Logger for output control
        self.logger = Logger(verbose)

        if monitor is not None and not isinstance(monitor, Monitor):
            raise TypeError(
                "monitor must be an instance of detpy.monitoring.Monitor"
            )

        self.monitor = monitor

        if self.db_writing_interval <= 0:
            raise ValueError("db_writing_interval must be positive")

        if self.db_writing_interval > self.nfe_max:
            self.logger.log(
                f"Warning: db_writing_interval ({self.db_writing_interval}) > "
                f"nfe_max ({self.nfe_max}). "
                "Data will be saved only at the end."
            )

        if params.max_nfe < params.population_size:
            raise ValueError(
                "max_nfe must be greater than or equal to population_size"
            )
        if len(self.lb) != params.dimension:
            raise ValueError(
                f"Lower bounds length ({len(self.lb)}) "
                f"must equal dimension ({params.dimension})."
            )

        if len(self.ub) != params.dimension:
            raise ValueError(
                f"Upper bounds length ({len(self.ub)}) "
                f"must equal dimension ({params.dimension})."
            )

        for i, (lb, ub) in enumerate(zip(self.lb, self.ub)):
            if lb >= ub:
                raise ValueError(
                    f"lb[{i}] must be < ub[{i}], got {lb} >= {ub}."
                )

        ParameterValidator.int_min(
            params.population_size,
            4,
            "Population size"
        )

        ParameterValidator.positive_int(
            params.dimension,
            "Dimension"
        )

        ParameterValidator.positive_int(
            params.max_nfe,
            "Max NFE"
        )

        self._initialize()

    def _calculate_population_diversity(self):
        population_array = np.array([
            member.get_chromosomes()
            for member in self._pop.members
        ], dtype=float)

        if population_array.size == 0:
            return 0.0

        if population_array.ndim != 2:
            return 0.0

        if len(population_array) <= 1:
            return 0.0

        bounds_range = np.asarray(self.ub, dtype=float) - np.asarray(
            self.lb,
            dtype=float
        )

        if np.any(bounds_range <= 0):
            return 0.0

        normalized_population = (
                                        population_array - np.asarray(self.lb, dtype=float)
                                ) / bounds_range

        coordinate_std = np.std(
            normalized_population,
            axis=0,
            ddof=0
        )

        return float(np.mean(coordinate_std))

    def _close_monitor(self):
        if self.monitor is None:
            return

        try:
            self.monitor.close()
        except Exception as e:
            self.logger.log(
                f"Monitoring close error: {e}"
            )

    def get_monitor_metrics(self) -> dict:
        """
        Return algorithm-specific metrics for monitoring.

        Algorithms can override this method to expose
        additional metrics to Monitor implementations.
        """
        return {}

    def _log_monitor_epoch(self, monitor_data: EpochData):
        if self.monitor is None:
            return

        try:
            self.monitor.log_epoch(
                monitor_data,
                metrics=self.get_monitor_metrics()
            )
        except Exception as e:
            self.logger.log(
                f"Monitoring error: {e}"
            )
    def _build_epoch_data(
            self,
            epoch,
            nfe,
            best_fitness,
            mean_fitness,
            std_fitness,
            epoch_time,
            population_fitnesses,
            evaluations_per_second=None
    ):
        return EpochData(
            algorithm_name=self.name,
            epoch=epoch,
            nfe=nfe,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            std_fitness=std_fitness,
            epoch_time=epoch_time,
            population_diversity=self._calculate_population_diversity(),
            population_min_fitness=min(population_fitnesses),
            population_max_fitness=max(population_fitnesses),
            evaluations_per_second=evaluations_per_second
        )

    @abstractmethod
    def next_epoch(self):
        pass

    @property
    def nfe(self) -> int:
        """Number of function evaluations performed so far."""
        return self._function.evaluation_count

    def _initialize(self):
        init_time = time.time()

        population = Population(
            lb=self.lb,
            ub=self.ub,
            arg_num=self.nr_of_args,
            size=self.population_size,
            optimization=self.optimization_type
        )

        population.generate_population()
        population.update_fitness_values(
            self._function.eval,
            self.parallel_processing
        )

        end_init_time = time.time()

        self._origin_pop = population
        self._pop = copy.deepcopy(population)

        # Creating table
        if self._database is not None:
            self._database.connect()

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            func_name = self._function.get_name()

            uuid_length = 8
            run_id = str(uuid.uuid4())[:uuid_length]

            table_name = get_table_name(
                func_name=func_name,
                alg_name=self.name,
                nr_of_args=self.nr_of_args,
                pop_size=self.population_size,
                timestamp=timestamp,
                run_id=run_id
            )

            self.database_table_name = self._database.create_table(table_name)
            self._database.close()
        else:
            self.database_table_name = None

        self._total_init_time = end_init_time - init_time

    def run(self):
        epoch_metrics = []
        best_fitness_values = []
        avg_fitness_values = []
        std_fitness_values = []

        # We store NFE values to plot them later
        nfe_numbers = []

        # Calculate metrics
        epoch_metric = MetricHelper.calculate_start_metrics(
            self._pop,
            self._total_init_time,
            self.nfe,
            self.log_population
        )

        epoch_metrics.append(epoch_metric)

        initial_best_member = self._pop.get_best_members(1)[0]

        best_fitness_values.append(
            initial_best_member.fitness_value
        )

        nfe_numbers.append(
            self.nfe
        )

        initial_fitnesses = [
            member.fitness_value
            for member in self._pop.members
        ]

        self._log_monitor_epoch(
            self._build_epoch_data(
                epoch=0,
                nfe=self.nfe,
                best_fitness=initial_best_member.fitness_value,
                mean_fitness=np.mean(initial_fitnesses),
                std_fitness=(
                    np.std(initial_fitnesses)
                    if len(initial_fitnesses) > 1
                    else 0.0
                ),
                epoch_time=self._total_init_time,
                population_fitnesses=initial_fitnesses
            )
        )

        total_start_time = time.time()

        # Index of the first metric that has not yet been saved
        last_saved_idx = 0

        # Next NFE threshold at which data should be saved.
        #
        # Example for db_writing_interval=1000:
        #   1000 -> first threshold
        #   2000 -> second threshold
        #   3000 -> third threshold
        #
        # Since NFE can jump over the exact value, e.g.
        # 882 -> 1156, the write happens at 1156.
        next_db_write_nfe = self.db_writing_interval
        previous_nfe = self.nfe
        # Epoch 0 - metrics after init
        if (
                self._database is not None
                and self.db_auto_write
                and epoch_metrics
        ):
            try:
                self.write_results_to_database(
                    [epoch_metrics[0]]
                )

                last_saved_idx = 1

            except Exception as e:
                self.logger.log(
                    "An unexpected error occurred while "
                    f"writing initial metrics to database: {e}"
                )

        bar_format = '{l_bar}{bar}{r_bar}\n'

        with tqdm(
                total=self.nfe_max,
                desc=f"{self.name}",
                unit="nfe",
                bar_format=bar_format
        ) as pbar:

            # We need to update the NFE counter here because
            # the population initialization requires NFE calculations.
            progress_difference = self._function.evaluation_count - pbar.n
            pbar.update(progress_difference)

            while self._function.evaluation_count < self.nfe_max:

                if self.additional_stop_criteria.should_stop(
                        self._function.evaluation_count,
                        self._epoch_number,
                        self._pop.get_best_members(1)[0]
                ):
                    tqdm.write(
                        Fore.RED +
                        "Stopping criterion reached." +
                        Style.RESET_ALL
                    )
                    break

                # Update progress bar
                progress_difference = (
                        self._function.evaluation_count - pbar.n
                )
                pbar.update(progress_difference)

                try:
                    epoch_start_time = time.time()

                    self._epoch_number += 1

                    # -----------------------------------------
                    # Execute next epoch
                    # -----------------------------------------

                    self.next_epoch()

                    # -----------------------------------------
                    # Current state AFTER epoch
                    # -----------------------------------------

                    current_nfe = self._function.evaluation_count

                    best_member = self._pop.get_best_members(1)[0]

                    best_fitness = best_member.fitness_value

                    # -----------------------------------------
                    # Epoch timing
                    # -----------------------------------------

                    epoch_time = time.time() - epoch_start_time

                    # -----------------------------------------
                    # Fitness statistics
                    # -----------------------------------------

                    population_fitnesses = [
                        member.fitness_value
                        for member in self._pop.members
                    ]

                    avg_fitness = float(
                        np.mean(population_fitnesses)
                    )

                    if len(population_fitnesses) > 1:
                        std_fitness = float(
                            np.std(population_fitnesses, ddof=0)
                        )
                    else:
                        std_fitness = 0.0

                    # -----------------------------------------
                    # Store values for plotting
                    # -----------------------------------------

                    best_fitness_values.append(
                        best_fitness
                    )

                    nfe_numbers.append(
                        current_nfe
                    )

                    avg_fitness_values.append(
                        avg_fitness
                    )

                    std_fitness_values.append(
                        std_fitness
                    )

                    # -----------------------------------------
                    # Metrics
                    # -----------------------------------------

                    epoch_metric = MetricHelper.calculate_metrics(
                        self._pop,
                        epoch_start_time,
                        self._epoch_number,
                        current_nfe,
                        self.log_population
                    )

                    epoch_metrics.append(
                        epoch_metric
                    )

                    # -----------------------------------------
                    # Performance
                    # -----------------------------------------

                    evaluations = (
                            current_nfe - previous_nfe
                    )

                    evaluations_per_second = (
                        evaluations / epoch_time
                        if epoch_time > 0
                        else 0.0
                    )

                    previous_nfe = current_nfe

                    # -----------------------------------------
                    # Console logger
                    # -----------------------------------------

                    self.logger.log(
                        f"NFE {current_nfe}/"
                        f"{self.nfe_max}, "
                        f"Best Fitness: {best_fitness}, "
                        f"Best Individual: "
                        f"{[member.real_value for member in best_member.chromosomes]}, "
                        f"Avg: {avg_fitness}, "
                        f"Std: {std_fitness}"
                    )

                    # -----------------------------------------
                    # Monitor
                    # -----------------------------------------

                    self._log_monitor_epoch(
                        self._build_epoch_data(
                            epoch=self._epoch_number,
                            nfe=current_nfe,
                            best_fitness=best_fitness,
                            mean_fitness=avg_fitness,
                            std_fitness=std_fitness,
                            epoch_time=epoch_time,
                            population_fitnesses=population_fitnesses,
                            evaluations_per_second=evaluations_per_second
                        )
                    )

                    # -----------------------------------------
                    # Database auto write
                    # -----------------------------------------

                    if (
                            self._database is not None
                            and self.db_auto_write
                            and current_nfe >= next_db_write_nfe
                    ):
                        try:
                            self.write_results_to_database(
                                [epoch_metrics[-1]]
                            )

                            last_saved_idx = len(epoch_metrics)

                            while current_nfe >= next_db_write_nfe:
                                next_db_write_nfe += self.db_writing_interval

                        except Exception as e:
                            self.logger.log(
                                "An unexpected error occurred while "
                                f"writing to the database: {e}"
                            )

                except Exception as e:
                    traceback.print_exc()

                    self.logger.log(
                        "An unexpected error occurred during calculation: "
                        f"{e}"
                    )

                    self._close_monitor()
                    return epoch_metrics

            # Ensure the progress bar finishes even if stopped early
            # due to additional stop criteria
            if pbar.n < pbar.total:
                pbar.update(pbar.total - pbar.n)

        end_time = time.time()

        execution_time = end_time - total_start_time

        self.logger.log(
            f"Function: {self._function.get_name()}, "
            f"Dimension: {self.nr_of_args}, "
            f"Execution time: {round(execution_time, 2)} seconds"
        )

        final_population_fitnesses = [
            member.fitness_value
            for member in self._pop.members
        ]

        avg_fitness = float(
            np.mean(final_population_fitnesses)
        )

        std_fitness = float(
            np.std(final_population_fitnesses, ddof=0)
        )

        best_solution = self._pop.get_best_members(1)[0]

        self.logger.log(
            f"Average Best Fitness: {avg_fitness}, "
            f"Standard Deviation of Fitness: {std_fitness}"
        )

        self.logger.log(
            f"Best Solution: {best_solution}"
        )

        # FINAL DATABASE WRITE
        # If db_auto_write=False:
        #     save the entire list.
        #
        # If db_auto_write=True:
        #     save only metrics that have not been saved yet.

        if self._database is not None:

            if not self.db_auto_write:
                # Save entire list
                try:
                    if not epoch_metrics:
                        self.logger.log("Warning: No metrics to save.")

                    filtered_metrics = []

                    if epoch_metrics:
                        filtered_metrics.append(epoch_metrics[0])

                    next_save_nfe = self.db_writing_interval

                    for metric in epoch_metrics[1:]:
                        if metric.nfe >= next_save_nfe:
                            filtered_metrics.append(metric)

                            while metric.nfe >= next_save_nfe:
                                next_save_nfe += self.db_writing_interval

                    self.write_results_to_database(filtered_metrics)

                except Exception as e:
                    self.logger.log(f'An unexpected error occurred while writing to the database: {e}')

            else:
                # Save remaining metrics that were collected after
                # the last automatic database write.
                try:
                    if last_saved_idx < len(epoch_metrics):
                        self.write_results_to_database(
                            epoch_metrics[last_saved_idx:]
                        )

                except Exception as e:
                    self.logger.log(
                        "An unexpected error occurred while writing "
                        f"remaining data to the database: {e}"
                    )

        result = AlgorithmResult(
            epoch_metrics=epoch_metrics,
            avg_fitness=avg_fitness,
            std_fitness=std_fitness,
            best_solution=best_solution
        )

        if self.show_plots:
            result.plot_results(
                nfe_numbers,
                best_fitness_values,
                avg_fitness_values,
                std_fitness_values,
                method_name=self.name
            )

        self._close_monitor()

        return result

    def write_results_to_database(self, results_data):
        self.logger.log(
            "Writing to Database..."
        )

        # Check if database is present
        if (
                self._database is None
                or self.database_table_name is None
        ):
            self.logger.log(
                "There is no database."
            )
            return

        # Nothing to save
        if not results_data:
            self.logger.log(
                "No new results to save."
            )
            return

        # Connect to database
        self._database.connect()

        try:
            # Inserting data into database
            formatted_best_individuals = format_individuals(
                results_data
            )

            self._database.insert_multiple_best_individuals(
                self.database_table_name,
                formatted_best_individuals
            )

        finally:
            # Always close the database connection
            self._database.close()
