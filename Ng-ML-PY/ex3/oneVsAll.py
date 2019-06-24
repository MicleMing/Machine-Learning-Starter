import scipy.optimize as opt
import lrCostFunction as lCF
import numpy as np


def one_vs_all(X, y, num_labels, lmd):
    # Some useful variables
    (m, n) = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n + 1))

    # Add ones to the X data 2D-array
    X = np.c_[np.ones(m), X]

    initail_theta = np.zeros((1, n + 1))

    for i in range(num_labels):
        print('Optimizing for handwritten number {}...'.format(i))

        costFunc = lCF.lr_cost_function

        res = opt.minimize(costFunc, initail_theta, method='CG', jac=True, options={
            'maxiter': 50}, args=(X, y == i, lmd))
        all_theta[i] = res.x

    print('Done')

    return all_theta
