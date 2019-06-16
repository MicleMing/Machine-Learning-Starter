import matplotlib.pyplot as plt
import numpy as np


def plot_data(X, y, show=True):
    x1 = X[y == 1]
    x2 = X[y == 0]

    plt.scatter(x1[:, 0], x1[:, 1], marker='+', label='admitted')
    plt.scatter(x2[:, 0], x2[:, 1], marker='.', label='Not admitted')
    plt.legend()
    if show == True:
        plt.show()
