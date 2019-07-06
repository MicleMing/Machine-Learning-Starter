import numpy as np


def poly_features(X, p):
    # You need to return the following variable correctly.
    X_poly = np.zeros((X.size, p))

    for i in range(p):
        for j in range(X.size):
            X_poly[j, i] = np.power(X[j], i + 1)

    return X_poly
