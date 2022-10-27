from math import exp

def epsilon_decay(decay_factor):
    def epsilon_update(epsilon, *args):
        return epsilon * (1 - decay_factor)

    return epsilon_update


def epsilon_fixed(epsilon_value):
    def epsilon_update(*args):
        return epsilon_value

    return epsilon_update


def epsilon_linear(epsilon_max, epsilon_min, steps):
    epsilon_step = (epsilon_max - epsilon_min) / steps

    def epsilon_update(epsilon, *args):
        return epsilon - epsilon_step

    return epsilon_update


def epsilon_exp(epsilon_min, epsilon_max, decay_factor):
    def epsilon_update(epsilon, step):
        return epsilon_min + (epsilon_max - epsilon_min) * exp(-decay_factor * step)

    return epsilon_update
