import numpy as np
import pytest
from pytest import approx
from scipy.stats import kendalltau
from gamma_correlation import gamma_corr
from gamma_correlation.tnorms import *
from gamma_correlation.fuzzy import *
import math

@pytest.fixture(autouse=True)
def set_random_seed():
    # seeds any random state in the tests, regardless where is defined
    np.random.seed(0)


def test_uncorrelated():
    ranking_a = [1, 2, 3, 4, 5]
    ranking_b = [5, 4, 3, 2, 1]

    assert gamma_corr(ranking_a, ranking_b)[0] == -1


def test_identical():
    first, second = generate_random_lists(6)
    assert gamma_corr(first, first)[0] == 1


def test_identical2():
    first, second = generate_random_lists(6)
    assert gamma_corr(first, first, weights=np.random.uniform(size=5))[0] == 1


def generate_random_lists(length):
    first = np.arange(1, length + 1)
    second = np.arange(1, length + 1)

    np.random.shuffle(first)
    np.random.shuffle(second)

    return first.tolist(), second.tolist()


def test_kendall():
    first, second = generate_random_lists(6)
    gamma = gamma_corr(first, second, weights="uniform", tnorm_type=prod)
    gamma_result = gamma[0]*(gamma[1]+gamma[2])
    tau, _ = kendalltau(first, second)
    tau_result = tau * math.comb(len(first), 2)
    assert np.isclose(gamma_result, tau_result, atol=1e-2), f"gamma_corr: {gamma_result}, kendalltau: {tau}"


def test_matrix():
    # ランダムなデータを生成
    rows, cols = 5, 5
    R_a = np.random.rand(rows, cols)
    R_b = np.random.rand(rows, cols)
    E_a = np.random.rand(rows, cols)
    E_b = np.random.rand(rows, cols)
    tnorm_type = prod

    triu_indices = np.triu_indices(rows)

    #vector
    C_matrix_vec = (np.vectorize(tnorm)(R_a[triu_indices], R_b[triu_indices], tnorm_type)
                    + np.vectorize(tnorm)(R_a.T[triu_indices], R_b.T[triu_indices], tnorm_type))
    D_matrix_vec = (np.vectorize(tnorm)(R_a[triu_indices], R_b.T[triu_indices], tnorm_type)
                    + np.vectorize(tnorm)(R_a.T[triu_indices], R_b[triu_indices], tnorm_type))
    T_matrix_vec = np.vectorize(conorm)(E_a[triu_indices], E_b[triu_indices], tnorm_type)

    print(C_matrix_vec, D_matrix_vec, T_matrix_vec)
    con_vec = np.sum(C_matrix_vec)
    dis_vec = np.sum(D_matrix_vec)
    tie_vec = np.sum(T_matrix_vec)

    #loop
    C_matrix_loop = np.zeros([rows, cols])
    D_matrix_loop = np.zeros([rows, cols])
    T_matrix_loop = np.zeros([rows, cols])

    for i in range(rows):
        for j in range(cols):
            C_matrix_loop[i, j] = tnorm(R_a[i, j], R_b[i, j], tnorm_type) + tnorm(R_a[j, i], R_b[j, i], tnorm_type)
            D_matrix_loop[i, j] = tnorm(R_a[i, j], R_b[j, i], tnorm_type) + tnorm(R_a[j, i], R_b[i, j], tnorm_type)
            T_matrix_loop[i, j] = conorm(E_a[i, j], E_b[i, j], tnorm_type)

    con_loop = np.sum(C_matrix_loop[triu_indices])
    dis_loop = np.sum(D_matrix_loop[triu_indices])
    tie_loop = np.sum(T_matrix_loop[triu_indices])

    assert np.isclose(con_vec, con_loop), f"con_vec ({con_vec}) does not match con_loop ({con_loop})"
    assert np.isclose(dis_vec, dis_loop), f"dis_vec ({dis_vec}) does not match dis_loop ({dis_loop})"
    assert np.isclose(tie_vec, tie_loop), f"tie_vec ({tie_vec}) does not match tie_loop ({tie_loop})"


def test_distance():
    weight = [1, 0.8, 0.5, 0.8, 1]
    assert sum_based_distance(2, 3, weight) == 0.8
    assert max_based_distance(2, 3, weight) == 0.8
    assert sum_based_distance(2, 4, weight) == 1
    assert max_based_distance(2, 3, weight) == 0.8


@pytest.mark.parametrize("mode,expected",
                         [("top", -0.5483870967),
                          ("bottom", -0.25),
                          ("top bottom", -0.5),
                          ("middle", -0.25),
                          ("top bottom exp", -0.5)])
def test_weights(mode, expected):
    n = 4
    a = np.arange(n) + 1
    ranking_a = np.random.permutation(a)
    ranking_b = np.random.permutation(a)

    assert approx(gamma_corr(ranking_a, ranking_b, weights=mode)[0]) == expected


# @pytest.mark.parametrize("func,expected",
#                          [(weight_agg_clamped_sum, -0.5555555555555556),
#                           (weight_agg_max, -0.5483870967741935)])
# def test_dists(func, expected):
#     n = 4
#     a = np.arange(n) + 1
#     ranking_a = np.random.permutation(a)
#     ranking_b = np.random.permutation(a)
#
#     assert approx(gamma_corr(ranking_a, ranking_b, weights="top", weight_agg=func)) == expected
