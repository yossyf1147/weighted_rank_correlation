from typing import Union, Optional
import math
from gamma_correlation.fuzzy import fuzzy_D
from gamma_correlation.tnorms import *
from gamma_correlation.weights import gen_weights


def gamma_corr(ranking_a: Union[list, np.ndarray], ranking_b: Union[list, np.ndarray], *,
               weights: Optional[Union[str, np.array]] = None, tnorm_type=prod):
    """
    :param ranking_a: First ranking
    :param ranking_b: Second ranking
    :param weights:
        If left empty weights will be set uniformly to 1.
        Weights between pairwise orderings. Must be one shorter than the length of the rankings.
        It can also be one of the following "uniform", "top", "bottom", "top bottom", "middle". Please refer to gen_weights for more detail
    :param tnorm_type: T-Norm function to use
    :return:
    """

    if len(ranking_a) != len(ranking_b):
        raise ValueError("not the same shape")

    rankings = np.array([ranking_a, ranking_b])
    n, rank_length = rankings.shape

    global weight_vec
    if weights is None:
        weight_vec = gen_weights("uniform", rank_length)
    if isinstance(weights, str):
        weight_vec = gen_weights(weights, rank_length)
    elif isinstance(weights, np.ndarray):
        weight_vec = weights  # type:np.array

    # upper triangle matrix to calculate all pairwise comparisons
    triu = np.triu_indices(rank_length, 1)

    def D_matrix_Calcuration(ranking: np.array) -> np.array:
        """
        :param ranking: 1 × n array of an ordering
        :return: n × n pairwise weight aggregations.
        """

        pair_indices = np.array(triu)
        # calculate pairwise rank positions
        rank_positions = ranking[pair_indices]

        # reshape the pairs back into a matrix
        matrix = np.zeros([rank_length, rank_length])

        # first we fill lower triangle with the inverse rank positions
        matrix[triu] = np.apply_along_axis(fuzzy_D, 0, np.flipud(rank_positions), weight=weight_vec)
        matrix = matrix.T

        # after transposing we fill the top triangle
        matrix[triu] = np.apply_along_axis(fuzzy_D, 0, rank_positions, weight=weight_vec)

        return matrix  # n × n

    # calculate all pairwise comparisons for all rankings. This considers the weights
    D_a, D_b = np.apply_along_axis(D_matrix_Calcuration, 1, rankings)  # rank_length × rank_length
    d_a, d_b = np.abs(D_a), np.abs(D_b)  # rank_length × rank_length
    R_a, R_b = np.maximum(D_a, 0), np.maximum(D_b, 0)  # rank_length × rank_length
    E_a, E_b = 1 - d_a, 1 - d_b  # rank_length × rank_length

    rows, cols = R_a.shape
    C_matrix = np.zeros([rows, cols])
    D_matrix = np.zeros([rows, cols])
    T_matrix = np.zeros([rows, cols])

    for i in range(rows):
        for j in range(cols):
            C_matrix[i, j] = T(R_a[i, j], R_b[i, j], tnorm_type) + T(R_a[j, i], R_b[j, i], tnorm_type)
            D_matrix[i, j] = T(R_a[i, j], R_b[j, i], tnorm_type) + T(R_a[j, i], R_b[i, j], tnorm_type)
            T_matrix[i, j] = conorm(E_a[i, j], E_b[i, j], tnorm_type)

    # print(C_matrix)
    # print(D_matrix)
    # print(T_matrix)

    con = np.sum(C_matrix[triu])
    dis = np.sum(D_matrix[triu])
    tie = np.sum(T_matrix[triu])

    # print("nC2: ", math.comb(rows, 2))
    # print("condistie: ", (con + dis + tie))

    try:
        return (con - dis) / (con + dis)
    except ZeroDivisionError:
        return np.nan


if __name__ == '__main__':
    first = [1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
    second = [3, 4, 2, 1, 6, 8, 8, 10, 10, 5]

    print("gamma: ", gamma_corr(first, second, weights="top", tnorm_type=hamacher))
