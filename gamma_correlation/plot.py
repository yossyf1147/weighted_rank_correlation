from gamma_correlation.weights import gen_weights, gen_beta_weights, gen_quadratic_weights, gen_yoshi_weights
from matplotlib import pyplot as plt
import cupy as cp
import numpy as np

def graph_quad_plot(a, b):
    weights = gen_quadratic_weights(a, b, 10)
    print(weights)
    plt.plot(np.linspace(0, 1, 9), weights)
    plt.xlabel('Index')
    plt.ylabel('Weight')
    plt.title('Quadratic Weights')
    plt.ylim(0, 1)
    plt.show()


def graph_beta_plot(flipped, a, b):
    weights = gen_beta_weights(flipped, a, b, 1000)
    plt.plot(np.linspace(0, 1, 999), weights)
    plt.xlabel('Index')
    plt.ylabel('Weight')
    plt.title('Quadratic Weights')
    plt.ylim(0, 1)
    plt.show()


# def graph_yoshi_plot(is_positive, point):
#     weights = gen_quadratic_weights(is_positive, point, 1000)
#     plt.plot(np.linspace(0.001, 0.999, 999), weights)
#     plt.xlabel('Index')
#     plt.ylabel('Weight')
#     plt.title('Quadratic Weights')
#     plt.ylim(0, 1)
#     plt.show()