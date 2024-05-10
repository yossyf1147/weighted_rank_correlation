import numpy as np
import scipy.special as sc
from scipy.stats import beta


def cropped_linspace(start, end, len_):
    return np.linspace(start, end, len_ + 1)[1:-1]


def gen_weights(mode, len_):

    match mode:
        case "uniform":
            return np.ones(len_ - 1)
        case "top":
            return cropped_linspace(1, 0, len_)
        case "bottom":
            return cropped_linspace(0, 1, len_)
        case "top bottom":
            return np.abs(cropped_linspace(1, -1, len_))
        case "middle":
            return 1 - np.abs(cropped_linspace(1, -1, len_))
        case 'top bottom exp':
            return 4 * (cropped_linspace(0, 1, len_) - 0.5) ** 2
        case _:
            raise AttributeError(f'mode "{mode}" not defined')


def gen_beta_weights(flipped: bool, alpha: float, beta_: float, length: int) -> np.ndarray:
    """
    Generate weights from Beta distribution.

    :param flipped:
    :param alpha: Alpha parameter of Beta distribution
    :param beta_: Beta parameter of Beta distribution
    :param length: Length of the weight vector
    :return: Array of weights generated from Beta distribution
    """
    x = cropped_linspace(0, 1, length)
    y = beta.pdf(x, alpha, beta_)
    y /= np.max(y)
    if flipped:
        return 1 - y
    else:
        return y


def gen_quadratic_weights(intercept: bool, x_intercept: float, length: int) -> np.ndarray:
    """
    Generate weights from a quadratic function.

    :param x_intercept:
    :param intercept:
    :param length: Length of the weight vector
    :return: Array of weights generated from the quadratic function
    """
    x = np.linspace(0.001, 0.999, length - 1)
    x_intercept = np.clip(x_intercept, 0, 1)
    gradient_a = 1 / (x_intercept * x_intercept)
    gradient_b = 1 / ((1 - x_intercept) * (1 - x_intercept))
    x_square = (x - x_intercept) * (x - x_intercept)
    if intercept:
        if x_intercept <= 0.5:
            weights = x_square * gradient_b
        else:
            weights = x_square * gradient_a
    else:
        if x_intercept <= 0.5:
            weights = -x_square * gradient_b + 1
        else:
            weights = -x_square * gradient_a + 1
    return weights


def gen_yoshi_weights(is_positive: bool, point: float, length: int) -> np.ndarray:
    point = np.clip(point, 0.0001, 0.9999)

    x = np.linspace(0, 1, length - 1)

    if is_positive:
        weights = np.where(x <= point, -x / point + 1, (x - 1) / (1 - point) + 1)
    else:
        weights = np.where(x <= point, x / point, - (x - 1) / (1 - point))

    return weights
