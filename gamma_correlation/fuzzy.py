import numpy as np
from numba import jit

# def fuzzy_E(a, b) -> float:
#     if a == b:
#         return 1
#     else:
#         return 1 - d(a, b)


def fuzzy_D(x: int, y: int, weight: np.ndarray, distance_func) -> float:
    if distance_func == "sum_based":
        return np.sign(y - x) * sum_based_distance(x, y, weight)
    elif distance_func == "max_based":
        return np.sign(y - x) * max_based_distance(x, y, weight)
    else:
        raise ValueError("Distance must be either sum_based or distance_based")


def sum_based_distance(x: int, y: int, weight: np.ndarray | list) -> float:
    """
    Calculate the distance between x and y based on the sum of distinguishability degrees.
    (15)
    Parameters:
    x (int): The first rank position.
    y (int): The second rank position.
    w (np.ndarray): The scaling function values for the rank positions.

    Returns:
    float: The distance between x and y.
    """
    if x == y:
        return 0
    min_pos = min(int(x) - 1, int(y) - 1)
    max_pos = max(int(x) - 1, int(y) - 1)
    distance = np.sum(weight[min_pos:max_pos])
    return min(1, distance)


def max_based_distance(x: int, y: int, weight: np.ndarray | list) -> float:
    """
    Calculate the distance between x and y based on the maximum of distinguishability degrees.
    (16)
    Parameters:
    x (int): The first rank position.
    y (int): The second rank position.
    w (np.ndarray): The scaling function values for the rank positions.

    Returns:
    float: The distance between x and y.
    """
    if x == y:
        return 0
    min_pos = min(int(x) - 1, int(y) - 1)
    max_pos = max(int(x) - 1, int(y) - 1)
    return np.max(weight[min_pos:max_pos])


def fuzzy_R(idx: np.array, weight: np.ndarray) -> float:
    #     if a == b:
    #         return 0
    #     else:
    #         if a < b:
    #             return d(a, b)
    #         else:
    #             return 0
    return weight[slice(int(idx[0]) - 1, int(idx[1]) - 1)].max(initial=0)
