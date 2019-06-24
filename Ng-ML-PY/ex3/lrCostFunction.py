import numpy as np
from sigmoid import *


def lr_cost_function(theta, X, y, lmd):
    m = y.size
    cost = 0
    grad = np.zeros(theta.shape)

    theta_for_reg = np.copy(theta)
    theta_for_reg[0] = 0

    sigmodiValue = sigmoid(X.dot(theta))
    yEq1 = ~y * np.log(sigmodiValue)
    yEq0 = (1 - y) * np.log(1 - sigmodiValue)

    reg = lmd * (theta_for_reg.dot(theta_for_reg))/(2 * m)
    cost = (1 / m) * np.sum(yEq1 - yEq0) + reg

    grad = (1 / m) * (sigmodiValue - y).dot(X) + lmd / m * theta_for_reg

    return cost, grad
