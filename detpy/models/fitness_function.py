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
        return self.function.eval(params)


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
        self._evaluation_count = 0

    def eval(self, *args, **kwargs) -> float:
        """
        Evaluate the wrapped fitness function and increment the evaluation count.

        :param args: Positional arguments for the fitness function.
        :param kwargs: Keyword arguments for the fitness function.
        :return: The result of the fitness function.
        """
        self._evaluation_count += 1
        return self._func.eval(*args, **kwargs)

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
        self._evaluation_count = 0

    def get_name(self) -> str:
        """
        Get the name of the wrapped fitness function.

        :return: The name of the fitness function.
        """
        return self._func.name
