import matplotlib.pyplot as plt
import numpy as np
from computeCost import compute_cost


def plot_data(X, y):
    plt.figure(0)
    plt.xlabel('Population of City in 10,000')
    plt.ylabel('Profit in $10,000')
    plt.plot(X, y, 'rx')
    plt.show()


def plot_J_history(X, y):
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

    for i in range(theta0_vals.size):
        for j in range(theta1_vals.size):
            theta = np.array([theta0_vals[i], theta1_vals[j]])
            t = compute_cost(X, y, theta)
            J_vals[i, j] = t

    theta_x, theta_y = np.meshgrid(theta0_vals, theta1_vals)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(theta_x, theta_y, J_vals)

    ax.set_xlabel(r'$\theta$0')
    ax.set_ylabel(r'$\theta$1')
    plt.show()
