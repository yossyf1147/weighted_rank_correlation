import numpy as np


# def fuzzy_E(a, b) -> float:
#     if a == b:
#         return 1
#     else:
#         return 1 - d(a, b)


def fuzzy_D(idx:np.array, weight: np.ndarray) -> float:
    if int(idx[0]) < int(idx[1]):
        return weight[slice(int(idx[0]) - 1, int(idx[1]) - 1)].max(initial=0)
    else:
        return -weight[slice(int(idx[1]) - 1, int(idx[0]) - 1)].max(initial=0)


def distance(idx: np.array, weight: np.ndarray) -> float:
    return weight[slice(min(int(idx[0]) - 1, int(idx[1]) - 1), max(int(idx[0]) - 1, int(idx[1]) - 1))].max(initial=0)


def fuzzy_R(idx: np.array, weight: np.ndarray) -> float:
    #     if a == b:
    #         return 0
    #     else:
    #         if a < b:
    #             return d(a, b)
    #         else:
    #             return 0
    return weight[slice(int(idx[0]) - 1, int(idx[1]) - 1)].max(initial=0)
