import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import plotData
from computeCost import compute_cost
from gradientDescent import gradient_descent

data = np.loadtxt('./data/ex1data1.txt', delimiter=',')

X = data[:, 0]
y = data[:, 1]
m = y.size

# plotData.plot_data(X, y)
X = np.c_[np.ones(m), X]

theta = np.zeros((2,))

iterations = 1500
alpha = 0.01


J = compute_cost(X, y, theta)


theta, J_history = gradient_descent(X, y, theta, alpha, iterations)

plotData.plot_J_history(X, y)
