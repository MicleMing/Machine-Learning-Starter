import numpy as np


def rand_initialization(l_in, l_out):
    # You need to return the following variable correctly
    w = np.zeros((l_out, 1 + l_in))

    epsilon_init = 0.12
    w = np.random.rand(l_out, l_in + 1) * 2 * epsilon_init - epsilon_init
    return w
