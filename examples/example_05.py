from detpy.DETAlgs.data.alg_data import MGDEData, EIDEData
from detpy.DETAlgs.eide import EIDE
from detpy.DETAlgs.mgde import MGDE
from detpy.models.enums.basevectorschema import BaseVectorSchema
from detpy.models.enums.boundary_constrain import BoundaryFixing
from detpy.models.enums.optimization import OptimizationType
from detpy.models.fitness_function import FitnessFunctionBase


class MyFitnessFunctionWrapper(FitnessFunctionBase):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.function = self._wrapped_fitness_func

    def _wrapped_fitness_func(self, solution):
        return MyFitnessFunc(solution)

    def eval(self, params):
        return self.function(params)


def MyFitnessFunc(solution):
    return solution[0] + solution[1]


if __name__ == "__main__":
    a = [0, 5]
    b = [5, 10]
    fitness_function = MyFitnessFunctionWrapper("Function name")

    params = EIDEData(
        epoch=10,
        population_size=10,
        dimension=len(a),
        lb=a,
        ub=b,
        optimization_type=OptimizationType.MINIMIZATION,
        base_vector_schema=BaseVectorSchema.RAND,
        boundary_constraints_fun=BoundaryFixing.RANDOM,
        function=fitness_function,
        log_population=True,
    )
    params.parallel_processing = ['thread', 5]

    default2 = EIDE(params, db_conn="Differential_evolution.db", db_auto_write=False)
    results = default2.run()
    default2.write_results_to_database(results.epoch_metrics)
