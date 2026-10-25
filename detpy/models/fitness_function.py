import threading
from typing import Callable
from abc import ABC, abstractmethod


class FitnessFunctionBase(ABC):
    def __init__(self):
        self.name = ""
        self.function = None

    @abstractmethod
    def eval(self, params):
        pass


class FitnessFunction(FitnessFunctionBase):
    def __init__(self, func: Callable[..., float], custom_name=None):
        super().__init__()
        self.name = func.__name__ if custom_name is None else custom_name
        self.function = func

    def eval(self, params):
        return self.function(*params)


class FitnessFunctionOpfunu(FitnessFunctionBase):
    def __init__(self, func_type, ndim, custom_name=None):
        super().__init__()
        self.name = func_type.__name__ if custom_name is None else custom_name
        self.function = func_type(ndim=ndim)

    def eval(self, params):
        return self.function.evaluate(params)


class BenchmarkFitnessFunction(FitnessFunctionBase):
    def __init__(self, function):
        super().__init__()
        self.instance = function
        self.name = function.name

    def eval(self, params):
        return self.instance.evaluate_func(params)


class FitnessFunctionWrapper(FitnessFunctionBase):
    """
    A wrapper for a fitness function that tracks the number of evaluations.
    """

    def __init__(self, func: FitnessFunctionBase):
        """
        Initialize the wrapper with a fitness function.

        :param func: The fitness function to wrap. Must be an instance of FitnessFunctionBase.
        :raises TypeError: If the provided function is not a FitnessFunctionBase instance.
        """
        if not isinstance(func, FitnessFunctionBase):
            raise TypeError("The wrapped function must be an instance of FitnessFunctionBase.")
        super().__init__()
        self._func = func
        self._inner_eval = func.eval  # Cache bound method for stable identity in pool caching
        self._evaluation_count = 0
        self._lock = threading.Lock()

    def eval(self, *args, **kwargs) -> float:
        """
        Evaluate the wrapped fitness function and increment the evaluation count.

        :param args: Positional arguments for the fitness function.
        :param kwargs: Keyword arguments for the fitness function.
        :return: The result of the fitness function.
        """
        with self._lock:
            self._evaluation_count += 1
        return self._func.eval(*args, **kwargs)

    def get_inner_eval(self):
        """
        Get the inner (unwrapped) eval function for use in process-based parallelism.
        The inner function does not increment the evaluation counter.

        :return: The eval method of the inner fitness function.
        """
        return self._inner_eval

    def add_evaluations(self, count: int) -> None:
        """
        Manually add to the evaluation counter.
        Used after process-based parallel evaluation where each process
        has its own copy of the function and cannot share the counter.

        :param count: Number of evaluations to add.
        """
        with self._lock:
            self._evaluation_count += count

    @property
    def evaluation_count(self) -> int:
        """
        Get the number of times the fitness function has been evaluated.

        :return: The evaluation count.
        """
        return self._evaluation_count

    def reset_evaluation_count(self) -> None:
        """
        Reset the evaluation count to zero.
        """
        with self._lock:
            self._evaluation_count = 0

    def get_name(self) -> str:
        """
        Get the name of the wrapped fitness function.

        :return: The name of the fitness function.
        """
        return self._func.name
