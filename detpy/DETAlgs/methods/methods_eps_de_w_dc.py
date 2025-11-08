def epsilon_dynamic_control(epoch_number: int, theta: int, epsilon_constrained: list, t_prime: int,
                            control_generations: int, initial_epsilon: float = 0.0):
    if epoch_number == 0:
        return sorted(epsilon_constrained)[theta]

    if t_prime >= control_generations:
        return 0

    return initial_epsilon * (1 - (t_prime / control_generations)) ** control_generations


def calculate_t_prime(epoch_number: int, epsilon_constrained: list, eta: int, control_generations: int,
                      penalty_power: int, t_prime: int = 0, epsilon_dynamic_control: float = 0.0,
                      initial_epsilon: float = 0.0):
    if epoch_number == 0:
        return 0

    sorted_epsilon_constrained = sorted(epsilon_constrained, reverse=True)
    eta_epsilon_constrained = sorted_epsilon_constrained[eta]
    if eta_epsilon_constrained >= epsilon_dynamic_control:
        return t_prime + 1

    epsilon_inverse = (1 - (eta_epsilon_constrained / initial_epsilon) ** (1 / penalty_power)) * control_generations
    if eta_epsilon_constrained < epsilon_dynamic_control and (t_prime + 2) >= epsilon_inverse:
        return t_prime + 2

    return 0.5 * (t_prime + 2) + 0.5 * epsilon_inverse