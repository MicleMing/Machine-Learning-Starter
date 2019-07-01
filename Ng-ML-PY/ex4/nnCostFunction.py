import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoid_gradient


def nn_cost_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd):
    theta1 = nn_params[0:hidden_layer_size *
                       (input_layer_size+1)].reshape((hidden_layer_size, input_layer_size+1))
    theta2 = nn_params[(hidden_layer_size*(input_layer_size+1))                       :].reshape((num_labels, hidden_layer_size+1))
    # Cost
    m = y.size
    a1 = np.c_[np.ones((m, 1)), X]
    z2 = a1.dot(theta1.T)
    a2 = np.c_[np.ones((np.size(z2, 0), 1)), sigmoid(z2)]
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)
    yt = np.zeros((m, num_labels))
    #  translate number to vector: 10 => [0.....1]
    yt[np.arange(m), y-1] = 1

    cost = np.sum(-yt * np.log(a3)-(1 - yt)*np.log(1 - a3))
    # add regularize item
    reg = lmd / (2 * m) * (np.sum(np.power(theta1[:, 1:], 2)) +
                           np.sum(np.power(theta2[:, 1:], 2)))
    cost = cost / m + reg

    # Backpropagation
    delta3 = a3 - yt

    z2WithBias = np.c_[np.ones((np.size(z2, 0), 1)), z2]
    delta2 = delta3.dot(theta2) * sigmoid_gradient(z2WithBias)
    theta2_grad = delta3.T.dot(a2)
    theta1_grad = delta2[:, 1:].T.dot(a1)

    theta2_grad = theta2_grad / m
    theta2_grad[:, 1:] = theta2_grad[:, 1:] + lmd / m * theta2[:, 1:]
    theta1_grad = theta1_grad / m
    theta1_grad[:, 1:] = theta1_grad[:, 1:] + lmd / m * theta1[:, 1:]
    grad = np.concatenate((theta1_grad.flatten(), theta2_grad.flatten()))

    return cost, grad
