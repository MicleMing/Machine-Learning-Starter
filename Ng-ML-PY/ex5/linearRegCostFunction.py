import numpy as np


def compute_cost(X, y, theta, lmd):

    m = y.size

    h = X.dot(theta)
    v = 1 / (2 * m) * np.sum(np.power(h - y, 2))
    reg = lmd / (2 * m) * np.sum(np.power(theta[1:], 2))

    return v + reg


def compute_grad(X, y, theta, lmd):
    m = y.size
    grad = np.zeros(theta.shape)

    h = X.dot(theta)

    grad[0] = 1 / m * (h - y).dot(X[:, 0])
    grad[1] = 1 / m * (h - y).dot(X[:, 1]) + lmd / m * \
        np.sum(np.power(theta[1:], 2))

    return grad


def linear_reg_cost_function(theta, X, y, lmd):
    m = y.size

    cost = compute_cost(X, y, theta, lmd)
    grad = compute_grad(X, y, theta, lmd)
    return cost, grad
