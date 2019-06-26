
import matplotlib.pyplot as plt
import numpy as np
from mapFeature import *
from plotData import *


def plot_decision_boundary(theta, X, y):
    plot_data(X[:, 1:3], y, False)

    # Only need two points to define a line, so choose two endpoints
    plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])

    # Calculate the decision boundary line
    plot_y = (-1/theta[2]) * (theta[1]*plot_x + theta[0])

    plt.plot(plot_x, plot_y)

    plt.legend(['Decision Boundary', 'Admitted', 'Not admitted'], loc=1)
    plt.axis([30, 100, 30, 100])
    plt.show()
