import copy
import time
import traceback
from abc import ABC, abstractmethod
from statistics import mean, stdev
from tqdm import tqdm
import numpy as np
import detpy
from detpy.database.database_connector import SQLiteConnector
from detpy.DETAlgs.data.alg_data import BaseData
from detpy.helpers.database_helper import get_table_name, format_individuals
from detpy.helpers.metric_helper import MetricHelper
from detpy.models.algorithm_result import AlgorithmResult
from detpy.models.fitness_function import FitnessFunctionBase
from detpy.models.population import Population
from detpy.helpers.logger import Logger


def example_function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    return (x1 - 1) ** 2 + (x2 - 2) ** 2 + (x3 - 3) ** 2 + (x4 - 4) ** 2 + (x5 - 5) ** 2 + \
           (x6 - 6) ** 2 + (x7 - 7) ** 2 + (x8 - 8) ** 2 + (x9 - 9) ** 2 + (x10 - 10) ** 2


class BaseAlg(ABC):
    def __init__(self, name, params: BaseData, db_conn=None, db_auto_write=False, verbose=True):
        self.name = name
        self._epoch_number = 0

        self._origin_pop = None
        self._pop = None
        self._total_init_time = None

        self.num_of_epochs = params.epoch
        self.population_size = params.population_size
        self.nr_of_args = params.dimension
        self.lb = params.lb
        self.ub = params.ub
        self.optimization_type = params.optimization_type
        self.boundary_constraints_fun = params.boundary_constraints_fun

        if params.function is None:
            self._function = detpy.FitnessFunction(func=example_function)
        else:
            self._function: FitnessFunctionBase = params.function

        self._database = SQLiteConnector(db_conn) if db_conn is not None else None
        self.db_auto_write = db_auto_write
        self.log_population = params.log_population
        self.parallel_processing = params.parallel_processing
        self.database_table_name = None
        self.db_writing_interval = 50

        # Use Logger for output control
        self.logger = Logger(verbose)
        self._initialize()

    @abstractmethod
    def next_epoch(self):
        pass

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
        population.update_fitness_values(self._function.eval, self.parallel_processing)
        end_init_time = time.time()

        self._origin_pop = population
        self._pop = copy.deepcopy(population)

        # Creating table
        self._database.connect()
        table_name = get_table_name(
            func_name="aa",
            alg_name=self.name,
            nr_of_args=self.nr_of_args,
            pop_size=self.population_size
        )
        self.database_table_name = self._database.create_table(table_name)
        self._database.close()
        self._total_init_time = end_init_time - init_time

    def run(self):
        epoch_metrics = []
        best_fitness_values = []
        avg_fitness_values = []
        std_fitness_values = []

        # Calculate metrics
        epoch_metric = MetricHelper.calculate_start_metrics(self._pop, self._total_init_time, self.log_population)
        epoch_metrics.append(epoch_metric)

        start_time = time.time()
        end_index = 0
        bar_format = '{l_bar}{bar}{r_bar}\n'

        for epoch in tqdm(range(self.num_of_epochs), desc=f"{self.name}", unit="epoch", bar_format=bar_format):
            best_member = self._pop.get_best_members(1)[0]
            best_fitness_values.append(best_member.fitness_value)

            try:
                start_time = time.time()
                self.next_epoch()
                # Calculate metrics
                epoch_metric = MetricHelper.calculate_metrics(self._pop, start_time, epoch, self.log_population)
                epoch_metrics.append(epoch_metric)

                avg_fitness = mean(member.fitness_value for member in self._pop.members)
                avg_fitness_values.append(avg_fitness)

                std_fitness = stdev(member.fitness_value for member in self._pop.members)
                std_fitness_values.append(std_fitness)

                self.logger.log(f"Epoch {epoch + 1}/{self.num_of_epochs}, Best Fitness: {best_member.fitness_value}, "
                                f"Best Individual: {[member.real_value for member in best_member.chromosomes]}, "
                                f"Avg: {avg_fitness}, Std: {std_fitness}")

                # Saving after each 50 epochs
                if epoch > 0 and epoch % self.db_writing_interval == 0:
                    end_index = epoch + 1
                    start_index = 0 if epoch == self.db_writing_interval else end_index - self.db_writing_interval
                    if self._database is not None and self.db_auto_write:
                        try:
                            self.write_results_to_database(epoch_metrics[start_index:end_index])
                        except:
                            self.logger.log('An unexpected error occurred while writing to the database.')
            except Exception as e:
                traceback.print_exc()
                self.logger.log(f'An unexpected error occurred during calculation: {e}')
                return epoch_metrics

        end_time = time.time()
        execution_time = end_time - start_time
        self.logger.log(f'Function: {self._function.name}, Dimension: {self.nr_of_args},'
                        f' Execution time: {round(execution_time, 2)} seconds')

        avg_fitness = np.mean(best_fitness_values)
        std_fitness = np.std(best_fitness_values)
        best_solution = self._pop.get_best_members(1)[0]

        self.logger.log(f"Average Best Fitness: {avg_fitness}, Standard Deviation of Fitness: {std_fitness}")
        self.logger.log(f"Best Solution: {best_solution}")

        # Writing to database
        if self._database is not None and not self.db_auto_write:
            try:
                self.write_results_to_database(epoch_metrics)
            except Exception as e:
                self.logger.log(f'An unexpected error occurred while writing to the database: {e}')
        elif self._database is not None and self.db_auto_write:
            try:
                self.write_results_to_database(epoch_metrics[end_index:])
            except Exception as e:
                self.logger.log(f'An unexpected error occurred while writing to the database: {e}')

        result = AlgorithmResult(
            epoch_metrics=epoch_metrics,
            avg_fitness=avg_fitness,
            std_fitness=std_fitness,
            best_solution=best_solution
        )

        result.plot_results(best_fitness_values, avg_fitness_values, std_fitness_values, self.num_of_epochs,
                            method_name=self.name)

        return result

    def write_results_to_database(self, results_data):
        self.logger.log(f'Writing to Database...')

        # Check if database is present
        if self._database is None or self.database_table_name is None:
            self.logger.log(f"There is no database.")
            return

        # Connect to database
        self._database.connect()

        # Inserting data into database
        formatted_best_individuals = format_individuals(results_data)
        self._database.insert_multiple_best_individuals(self.database_table_name, formatted_best_individuals)

        self._database.close()
