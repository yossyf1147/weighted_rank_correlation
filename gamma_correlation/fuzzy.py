import numpy as np

# def fuzzy_E(a, b) -> float:
#     if a == b:
#         return 1
#     else:
#         return 1 - d(a, b)
#
#
# def fuzzy_R(a, b) -> float:
#     if a == b:
#         return 0
#     else:
#         if a < b:
#             return d(a, b)
#         else:
#             return 0


def d(weights: np.ndarray) -> int:
    """
    Maximum of weights slice

    :param weights: distance weight
    :return:
    """
    return weights.max(initial=0)
