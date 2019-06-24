import numpy as np
from sigmoid import *


def predict_one_vs_all(all_theta, X):
    m = X.shape[0]  # Number of training examples

    # Add ones to the X data matrix
    X = np.vstack((np.ones(m), X.T)).T

    return np.argmax(sigmoid(np.dot(all_theta, X.T)), axis=0)
