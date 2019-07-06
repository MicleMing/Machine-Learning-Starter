import numpy as np

from trainLinearReg import train_linear_reg
from linearRegCostFunction import linear_reg_cost_function


def learning_curve(X, y, Xval, yval, lmd):
    # Number of training examples
    m = X.shape[0]

    # You need to return these values correctly
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(m):
        theta = train_linear_reg(X[0:i+1, :], y[0:i+1], lmd)
        error_train[i], _ = linear_reg_cost_function(
            theta, X[0:i+1, :], y[0:i+1], lmd)
        error_val[i], _ = linear_reg_cost_function(theta, Xval, yval, 0)

    return error_train, error_val
