import numpy as np


def prod(a: float, b: float):
    return a * b


def luka(a: float, b: float):
    """
    Łukasiewicz t-norm

    :param a:
    :param b:
    :return:
    """
    return np.maximum(a + b - 1, 0)


def drastic(a: float, b: float):
    return 0


def nilpotent(a: float, b: float):
    if a + b > 1:
        return min(a, b)
    else:
        return 0


def hamacher(a: float, b: float):
    if a == b == 0:
        return 0
    else:
        return a * b / (a + b - a * b)


def T(a: float, b: float, tnorm_type: callable):
    if a == 1:
        return b
    if b == 1:
        return a
    if a > b:
        return T(b, a, tnorm_type)
    try:
        return tnorm_type(a, b)
    except Exception as e:
        raise ValueError("Invalid t-norm") from e


def conorm(a: float, b: float, tnorm_type: callable):
    return 1 - T(1 - a, 1 - b, tnorm_type)
