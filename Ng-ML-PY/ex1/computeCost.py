import numpy as np


def compute_cost(X, y, theta):
    m = y.size
    prediction = X.dot(theta) - y
    sqr = np.power(prediction, 2)
    cost = (1 / (2 * m)) * np.sum(sqr)
    return cost
