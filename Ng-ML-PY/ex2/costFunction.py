import numpy as np
from sigmoid import *


def cost_function(theta, X, y):
    m = y.size
    cost = 0
    grad = np.zeros(theta.shape)

    item1 = -y * np.log(sigmoid(X.dot(theta)))
    item2 = (1 - y) * np.log(1 - sigmoid(X.dot(theta)))

    cost = (1 / m) * np.sum(item1 - item2)

    grad = (1 / m) * ((sigmoid(X.dot(theta)) - y).dot(X))

    return cost, grad
