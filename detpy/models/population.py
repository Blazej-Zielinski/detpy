import concurrent.futures
import numpy as np
from detpy.models.enums.optimization import OptimizationType
from detpy.models.member import Member


_worker_eval_func = None


def _init_process_worker(eval_func):
    """Initializer for process pool workers."""
    global _worker_eval_func
    _worker_eval_func = eval_func


def _evaluate_in_worker(chromosomes):
    """Evaluate chromosomes using the worker-local eval function."""
    return _worker_eval_func(chromosomes)


_thread_pool = None
_thread_pool_workers = 0

_process_pool = None
_process_pool_workers = 0
_process_pool_func = None


def _get_thread_pool(num_workers):
    """Return a cached ThreadPoolExecutor, recreating only if worker count changed."""
    global _thread_pool, _thread_pool_workers
    if _thread_pool is None or _thread_pool_workers != num_workers:
        if _thread_pool is not None:
            _thread_pool.shutdown(wait=False)
        _thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        _thread_pool_workers = num_workers
    return _thread_pool


def _get_process_pool(num_workers, eval_func):
    """Return a cached ProcessPoolExecutor, recreating on worker count or function change."""
    global _process_pool, _process_pool_workers, _process_pool_func
    if (_process_pool is None
            or _process_pool_workers != num_workers
            or _process_pool_func is not eval_func):
        if _process_pool is not None:
            _process_pool.shutdown(wait=True)
        _process_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_process_worker,
            initargs=(eval_func,),
        )
        _process_pool_workers = num_workers
        _process_pool_func = eval_func
    return _process_pool


class Population:
    def __init__(self, lb, ub, arg_num, size, optimization: OptimizationType):
        self.size = size
        self.members = None
        self.optimization = optimization
        self.lb = lb
        self.ub = ub
        self.arg_num = arg_num

    def generate_population(self):
        self.members = np.array([Member(self.lb, self.ub, self.arg_num) for _ in range(self.size)])

    @classmethod
    def with_new_members(cls, population, new_members):
        """
        Create a new Population instance with the same attributes as the given population,
        but with a new set of members.

        Parameters:
        - population (Population): The existing population to copy attributes from.
        - new_members (list): The new members to populate the new population.

        Returns:
        - Population: A new Population instance with the updated members.
        """
        new_population = cls(
            lb=population.lb,
            ub=population.ub,
            arg_num=population.arg_num,
            size=population.size,
            optimization=population.optimization
        )
        new_population.members = np.array(new_members)
        return new_population

    @staticmethod
    def calculate_fitness(member, fitness_fun):
        args = member.get_chromosomes()
        return fitness_fun(args)

    def calculate_member(self, index, fitness_fun):
        member = self.members[index]
        member.fitness_value = self.calculate_fitness(member, fitness_fun)

    def update_fitness_values(self, fitness_fun, parallel_processing=None):
        """
        Update fitness values for all members in the population.

        Parameters:
        - fitness_fun: The fitness evaluation function (typically FitnessFunctionWrapper.eval).
        - parallel_processing: Configuration for parallel execution.
            - None: Sequential execution.
            - ['thread', N]: Use N threads via ThreadPoolExecutor.
            - ['process', N]: Use N processes via ProcessPoolExecutor.
        """
        if parallel_processing is None:
            for idx in range(len(self.members)):
                self.calculate_member(idx, fitness_fun)
        elif parallel_processing[0] == "process":
            self._update_fitness_values_process(fitness_fun, parallel_processing[1])
        else:
            self._update_fitness_values_thread(fitness_fun, parallel_processing[1])

    def _update_fitness_values_thread(self, fitness_fun, num_workers):
        """
        Evaluate fitness using a cached ThreadPoolExecutor with batched tasks.
        """
        from detpy.models.fitness_function import FitnessFunctionWrapper
        is_wrapper = False
        if hasattr(fitness_fun, '__self__') and isinstance(fitness_fun.__self__, FitnessFunctionWrapper):
            inner_eval = fitness_fun.__self__.get_inner_eval()
            wrapper = fitness_fun.__self__
            is_wrapper = True
        else:
            inner_eval = fitness_fun

        pool = _get_thread_pool(num_workers)
        num_members = len(self.members)
        members = self.members
        batch_size = num_members // num_workers
        remainder = num_members % num_workers

        def evaluate_batch(batch_info):
            start, end = batch_info
            eval_fn = inner_eval
            for idx in range(start, end):
                member = members[idx]
                member.fitness_value = eval_fn(member.get_chromosomes())

        batches = []
        offset = 0
        for i in range(num_workers):
            size = batch_size + (1 if i < remainder else 0)
            batches.append((offset, offset + size))
            offset += size

        futures = [pool.submit(evaluate_batch, b) for b in batches]
        for f in futures:
            f.result()
        if is_wrapper:
            wrapper.add_evaluations(num_members)

    def _update_fitness_values_process(self, fitness_fun, num_workers):
        """Evaluate fitness using a cached ProcessPoolExecutor with initializer pattern."""
        from detpy.models.fitness_function import FitnessFunctionWrapper

        is_wrapper = False
        if hasattr(fitness_fun, '__self__') and isinstance(fitness_fun.__self__, FitnessFunctionWrapper):
            inner_eval = fitness_fun.__self__.get_inner_eval()
            wrapper = fitness_fun.__self__
            is_wrapper = True
        else:
            inner_eval = fitness_fun

        num_members = len(self.members)
        chromosomes_data = [self.members[i].get_chromosomes() for i in range(num_members)]
        pool = _get_process_pool(num_workers, inner_eval)
        chunksize = max(1, num_members // (num_workers * 2))
        fitness_values = list(pool.map(
            _evaluate_in_worker, chromosomes_data, chunksize=chunksize,
        ))
        for i, fv in enumerate(fitness_values):
            self.members[i].fitness_value = fv
        if is_wrapper:
            wrapper.add_evaluations(num_members)

    def get_best_members(self, nr_of_members):
        if self.optimization == OptimizationType.MINIMIZATION:
            sorted_indices = np.argsort([member.fitness_value for member in self.members])
        else:
            sorted_indices = np.argsort([member.fitness_value for member in self.members])[::-1]
        sorted_array = self.members[sorted_indices]
        return sorted_array[:nr_of_members]

    def mean(self):
        return np.mean([member.fitness_value for member in self.members])

    def std(self):
        return np.std([member.fitness_value for member in self.members])

    def resize(self, new_size: int):
        """
        Resize the population to the new size. If the new size is smaller, truncate the members array by removing the
        worst members. If the new size is larger, add new members with random chromosomes.

        Parameters:
        new_size (int): The new size of the population.
        """
        if new_size < self.size:
            if self.optimization == OptimizationType.MINIMIZATION:
                sorted_indices = np.argsort([member.fitness_value for member in self.members])
            else:
                sorted_indices = np.argsort([member.fitness_value for member in self.members])[::-1]
            self.members = self.members[sorted_indices[:new_size]]
        elif new_size > self.size:
            additional_members = [
                Member(
                    np.random.uniform(self.lb, self.ub, self.arg_num)
                )
                for _ in range(new_size - self.size)
            ]
            self.members = np.concatenate((self.members, additional_members))

        self.size = new_size

    def __str__(self, population_label=""):
        output = f"Population{population_label}:"
        for m in self.members:
            output += f"\n{str(m)}"

        return output
