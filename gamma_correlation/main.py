from typing import Union, Optional
import math
from fuzzy import *
from gamma_correlation.tnorms import *
from gamma_correlation.weights import *


def gamma_corr(ranking_a: Union[list, np.ndarray], ranking_b: Union[list, np.ndarray], *,
               weights: Optional[Union[str, np.array]] = None, tnorm=prod):
    """
    :param ranking_a: First ranking
    :param ranking_b: Second ranking
    :param weights:
        If left empty weights will be set uniformly to 1.
        Weights between pairwise orderings. Must be one shorter than the length of the rankings.
        It can also be one of the following "uniform", "top", "bottom", "top bottom", "middle". Please refer to gen_weights for more detail
    :param tnorm: T-Norm function to use
    :param weight_agg: Weight aggregation to use
    :return:
    """
    rankings = np.array([ranking_a, ranking_b])
    n, rank_length = rankings.shape

    if weights is None:
        weight_vec = gen_weights("uniform", rank_length)
    if isinstance(weights, str):
        weight_vec = gen_weights(weights, rank_length)
    elif isinstance(weights, np.ndarray):
        weight_vec = weights  # type:np.array

    def matrix_Calcuration(ranking: np.array, relation:str) -> np.array:
        """
        :param relation: String indicating the type of relation ('R' or 'd')
        :param ranking: 1 × n array of an ordering
        :return: n × n pairwise weight aggregations.
        """

        def distance(idx):
            slice_object = slice(min(int(idx[0]) - 1, int(idx[1]) - 1), max(int(idx[0]) - 1, int(idx[1]) - 1))
            weight_vector = weight_vec[slice_object]
            return d(weight_vector)

        def R(idx):
            slice_object = slice(int(idx[0]) - 1, int(idx[1]) - 1)
            weight_vector = weight_vec[slice_object]
            return d(weight_vector)

        if relation == 'R':
            relation_function = R
        elif relation == 'd':
            relation_function = distance
        else:
            raise ValueError("Invalid relation type. Must be 'R' or 'd'.")

        # upper triangle matrix to calculate all pairwise comparisons
        triu = np.triu_indices(rank_length, 1)
        pair_indices = np.array(triu)
        # calculate pairwise rank positions
        rank_positions = ranking[pair_indices]

        # calculate weight slices and aggregate, return aij and aji
        # reshape the pairs back into a matrix
        matrix = np.zeros([rank_length, rank_length])
        # first we fill lower triangle with the inverse rank positions
        matrix[triu] = np.apply_along_axis(relation_function, 0, np.flipud(rank_positions))
        matrix = matrix.T
        # after transposing we fill the top triangle
        matrix[triu] = np.apply_along_axis(relation_function, 0, rank_positions)

        return matrix  # n × n

    # calculate all pairwise comparisons for all rankings. This considers the weights
    d_a, d_b = np.apply_along_axis(matrix_Calcuration, 1, rankings, relation="d")  # rank_length × rank_length
    R_a, R_b = np.apply_along_axis(matrix_Calcuration, 1, rankings, relation="R")  # rank_length × rank_length
    E_a, E_b = 1 - d_a, 1 - d_b


    rows, cols = R_a.shape
    C_matrix = np.zeros([rows, cols])
    D_matrix = np.zeros([rows, cols])
    T_matrix = np.zeros([rows, cols])

    for i in range(rows):
        for j in range(cols):
            C_matrix[i, j] = T(R_a[i, j], R_b[i, j], tnorm) + T(R_a[j, i], R_b[j, i], tnorm)
            D_matrix[i, j] = T(R_a[i, j], R_b[j, i], tnorm) + T(R_a[j, i], R_b[i, j], tnorm)
            T_matrix[i, j] = conorm(E_a[i, j], E_b[i, j], tnorm)

    con = C_matrix.sum()
    dis = D_matrix.sum()
    tie = T_matrix.sum()
    print(math.comb(rows,2))
    print("condistie = nC2?", (con + dis + tie))

    try:
        return (con - dis) / (con + dis)
    except ZeroDivisionError:
        return np.nan


if __name__ == '__main__':
    first = [1, 3, 1, 4]
    second = [1, 1, 2, 1]

    print("gamma: ", gamma_corr(first, second, weights="top", tnorm=hamacher))
