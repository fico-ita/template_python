import random

import numpy as np


def get_uniform_noneg(size):
    """Generate initial uniform weights.

    Args:
        size (int): Number of weights to generate.

    Returns:
        ndarray: Array of initial uniform weights.
    """
    aux = [random.uniform(0, 2 / size) for _ in range(size)]
    initial_weights = np.array(aux)  # Start with all weights set to 0
    return initial_weights


def get_uniform_posneg(size):
    """Generate initial uniform weights.

    Args:
        size (int): Number of weights to generate.

    Returns:
        ndarray: Array of initial uniform weights.
    """
    aux = [random.uniform(-2 / size, 2 / size) for _ in range(size)]
    initial_weights = np.array(aux)  # Start with all weights set to 0
    return initial_weights
