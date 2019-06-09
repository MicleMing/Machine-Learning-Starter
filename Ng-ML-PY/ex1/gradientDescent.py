import numpy as np
from computeCost import compute_cost


def gradient_descent(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros((num_iters))

    for i in range(0, num_iters):
        prediction = X.dot(theta) - y
        delta = prediction.dot(X)
        theta = theta - alpha * (1 / m) * delta
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history
