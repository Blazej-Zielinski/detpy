import random


def eide_adopt_parameters(crossover_rate_min: float, crossover_rate_max: float, curr_generation: int,
                          max_generation: int) -> tuple[float, float]:
    new_f = random.uniform(0, 0.6)
    new_cr = crossover_rate_min + (crossover_rate_max - crossover_rate_min) * curr_generation / max_generation

    return new_f, new_cr
