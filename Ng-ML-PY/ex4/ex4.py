import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import scipy.optimize as opt

plt.ion()

# Setup the parameters you will use for this part of the exercise
input_layer_size = 400  # 20x20 input images of Digits
hidden_layer_size = 25  # 25 hidden layers
num_labels = 10         # 10 labels, from 0 to 9 Note that we have mapped "0" to label 10
