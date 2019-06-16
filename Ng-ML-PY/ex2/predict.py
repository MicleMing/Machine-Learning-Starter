import numpy as np
from sigmoid import *


def predict(theta, X):
    m = X.shape[0]
    p = np.zeros(m)

    prob = sigmoid(X.dot(theta))
    p = prob > 0.5
    return p
