from typing import Union, Optional
from gamma_correlation.fuzzy import *
from gamma_correlation.tnorms import *
from gamma_correlation.plot import *
import pandas as pd
import cupy as cp


def sequential_D_matrix_Calculation_gpu(ranking: cp.ndarray, weight_vec: cp.ndarray, distance_func) -> cp.ndarray:
    rank_length = len(ranking)
    matrix = cp.zeros((rank_length, rank_length))
    triu_indices = cp.triu_indices(rank_length, 1)

    for k in range(len(triu_indices[0])):
        i, j = triu_indices[0][k], triu_indices[1][k]
        matrix[i, j] = fuzzy_D(ranking[i], ranking[j], weight_vec, distance_func)
        matrix[j, i] = -matrix[i, j]
    return matrix

def gamma_corr(ranking_x: Union[list, cp.array, pd.core.series.Series], ranking_y: Union[list, cp.array, pd.core.series.Series], *,
               weights: Optional[Union[str, cp.array]] = "uniform", tnorm_type=luka, distance_func="max_based"):
    """
    :param distance_func:
    :param ranking_x: First ranking
    :param ranking_y: Second ranking
    :param weights:
        If left empty weights will be set uniformly to 1.
        Weights between pairwise orderings. Must be one shorter than the length of the rankings.
        It can also be one of the following "uniform", "top", "bottom", "top bottom", "middle". Please refer to gen_weights for more detail
    :param tnorm_type: T-Norm function to use
    :return:
    """
    # if not isinstance(ranking_x, (list, np.ndarray, pd.core.series.Series)) or not isinstance(ranking_y, (list, np.ndarray, pd.core.series.Series)):
    #     raise ValueError("Input must be a list, a NumPy array or pd.core.series.Series:", type(ranking_x))

    if isinstance(ranking_x, pd.core.series.Series):
        ranking_x = ranking_x.tolist()
        ranking_y = ranking_y.tolist()
    if isinstance(ranking_x, list):
        ranking_x = cp.array(ranking_x)
        ranking_y = cp.array(ranking_y)

    if len(ranking_x) != len(ranking_y):
        raise ValueError(ranking_x, ranking_y, "not the same shape")

    rank_length = len(ranking_x)

    if not cp.all((1 <= ranking_x) & (ranking_x <= rank_length)):
        raise ValueError("Elements of ranking_x must be within the range from 1 to the length of ranking_x")

    if not cp.all((1 <= ranking_y) & (ranking_y <= rank_length)):
        raise ValueError("Elements of ranking_y must be within the range from 1 to the length of ranking_y")

    if isinstance(weights, str):
        weight_vec = gen_weights(weights, rank_length)
    elif isinstance(weights, cp.ndarray):
        weight_vec = weights  # type:cp.array
    elif isinstance(weights, tuple) and len(weights) == 2:
        alpha, beta_val = weights
        weight_vec = gen_quadratic_weights(alpha, beta_val, rank_length)
    elif isinstance(weights, tuple) and len(weights) == 3:
        flipped, alpha, beta_val = weights
        weight_vec = gen_beta_weights(flipped, alpha, beta_val, rank_length)
    else:
        raise ValueError("Invalid weights format")

    if not len(weight_vec) + 1 == rank_length:
        raise ValueError("Invalid ranking length")

    D_x = sequential_D_matrix_Calculation_gpu(ranking_x, weight_vec, distance_func)
    D_y = sequential_D_matrix_Calculation_gpu(ranking_y, weight_vec, distance_func)

    d_x, d_y = cp.abs(D_x), cp.abs(D_y)
    R_x, R_y = cp.maximum(D_x, 0), cp.maximum(D_y, 0)

    if tnorm_type == luka or distance_func == "max_based":  #What else could hold the fuzzy logic?
        E_x, E_y = 1 - d_y, 1 - d_y
    else:
        raise ValueError("This condition does not satisfy the transitivity for fuzzy logic")


    triu_indices = cp.triu_indices(rank_length, 1)

    def elementwise_tnorm(a, b, tnorm_type):
        return cp.array([tnorm(ai, bi, tnorm_type) for ai, bi in zip(a, b)])

    def elementwise_conorm(a, b, tnorm_type):
        return cp.array([conorm(ai, bi, tnorm_type) for ai, bi in zip(a, b)])

    C_matrix_vec = elementwise_tnorm(R_x[triu_indices], R_y[triu_indices], tnorm_type) + elementwise_tnorm(R_x.T[triu_indices], R_y.T[triu_indices], tnorm_type)
    D_matrix_vec = elementwise_tnorm(R_x[triu_indices], R_y.T[triu_indices], tnorm_type) + elementwise_tnorm(R_x.T[triu_indices], R_y[triu_indices], tnorm_type)
    T_matrix_vec = elementwise_conorm(E_x[triu_indices], E_y[triu_indices], tnorm_type)


    con = cp.sum(C_matrix_vec)
    dis = cp.sum(D_matrix_vec)
    tie = cp.sum(T_matrix_vec)

    if con + dis == 0:
        gamma = 0
    else:
        gamma = (con - dis) / (con + dis)

    return gamma


if __name__ == '__main__':
    first = [1, 1, 3, 4, 5]
    second = [4, 5, 1, 2, 3]
    print("gamma: ", gamma_corr(first, second))


