import numpy as np

def prod(a, b):
    return a * b

def luka(a, b):
    """
    Łukasiewicz t-norm

    :param a:
    :param b:
    :return:
    """
    return np.maximum(a + b - 1, 0)

def drastic(a,b):
    return 0

def nilpotent(a,b):
    if a+b>1:
        return min(a,b)
    else:
        return 0

def hamacher(a,b):
    if a==b==0:
        return 0
    else:
        return a*b/(a+b-a*b)

def T(a, b, prominent):
    if a == 1:
        return b
    if b == 1:
        return a
    if a > b:
        return T(b, a, prominent)
    try:
        return prominent(a, b)
    except Exception as e:
        raise ValueError("Invalid t-norm") from e

def conorm(a, b, prominent):
    return 1 - T(1 - a, 1 - b, prominent)