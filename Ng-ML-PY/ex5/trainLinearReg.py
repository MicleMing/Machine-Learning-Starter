import numpy as np
import scipy.optimize as opt
from linearRegCostFunction import linear_reg_cost_function


def train_linear_reg(X, y, lmd):
    initial_theta = np.ones(X.shape[1])

    def cost_func(t):
        return linear_reg_cost_function(t, X, y, lmd)[0]

    def grad_func(t):
        return linear_reg_cost_function(t, X, y, lmd)[1]

    theta, *unused = opt.fmin_cg(cost_func, initial_theta, grad_func, maxiter=200, disp=False,
                                 full_output=True)

    return theta
