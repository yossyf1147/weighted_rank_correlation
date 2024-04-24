import numpy as np
import scipy.special as sc
from scipy.stats import beta

def gen_weights(mode, len_):
    def cropped_linspace(start, end):
        return np.linspace(start, end, len_ + 1)[1:-1]

    match mode:
        case "uniform":
            return np.ones(len_ - 1)
        case "top":
            return cropped_linspace(1, 0)
        case "bottom":
            return cropped_linspace(0, 1)
        case "top bottom":
            return np.abs(cropped_linspace(1, -1))
        case "middle":
            return 1 - np.abs(cropped_linspace(1, -1))
        case 'top bottom exp':
            return 4 * (cropped_linspace(0, 1) - 0.5) ** 2
        case _:
            raise AttributeError(f'mode "{mode}" not defined')


def gen_beta_weights(alpha: float, beta_: float, length: int) -> np.ndarray:
    """
    Generate weights from Beta distribution.

    :param alpha: Alpha parameter of Beta distribution
    :param beta_: Beta parameter of Beta distribution
    :param length: Length of the weight vector
    :return: Array of weights generated from Beta distribution
    """
    x = np.linspace(0.01, 0.99, length)
    y = beta.pdf(x, alpha, beta_)
    y /= np.sum(y)  # Normalize weights to sum up to 1
    # print(y)
    return y