
def calculate_best_member_count(population_size: int, p: float = 0.2) -> int:
    """
    Calculate the number of best members to select based on the population size and a given proportion p.
    Args:
        population_size (int):  The size of the population.
        p (float):  The proportion of the population to select as best members (default is 0.2).

    Returns: The number of best members to select.

    """
    return int(p * population_size)

