import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import scipy.optimize as opt
from displayData import display_data
from nnCostFunction import nn_cost_function
from sigmoidGradient import sigmoid_gradient
from randInitializeWeights import rand_initialization
from checkNNGradients import check_nn_gradients
from predict import predict

plt.ion()

# Setup the parameters you will use for this part of the exercise
input_layer_size = 400  # 20x20 input images of Digits
hidden_layer_size = 25  # 25 hidden layers
num_labels = 10         # 10 labels, from 0 to 9 Note that we have mapped "0" to label 10

# ===================== Part 1: Loading and Visualizing Data =====================
# We start the exercise by first loading and visualizing the dataset.
# You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

data = scio.loadmat('./data/ex4data1.mat')
X = data['X']
y = data['y'].flatten()
m = y.size

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
selected = X[rand_indices[0:100], :]

# display_data(selected)

# input('Program paused. Press ENTER to continue')


data = scio.loadmat('./data/ex4weights.mat')
theta1 = data['Theta1']
theta2 = data['Theta2']
nn_params = np.concatenate([theta1.flatten(), theta2.flatten()])

cost, grad = nn_cost_function(
    nn_params,
    input_layer_size,
    hidden_layer_size,
    num_labels,
    X,
    y,
    0
)
print(
    'Cost at parameters (loaded from ex4weights): {:0.6f}\n(This value should be about 0.287629)'.format(cost))

input('Program paused. Press ENTER to continue')

lmd = 1

cost, grad = nn_cost_function(
    nn_params,
    input_layer_size,
    hidden_layer_size,
    num_labels,
    X,
    y,
    lmd
)

print(
    'Cost at parameters (loaded from ex4weights): {:0.6f}\n(This value should be about 0.383770)'.format(cost))

input('Program paused. Press ENTER to continue')

print('Evaluating sigmoid gradient ...')

g = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))

print('Sigmoid gradient evaluated at [-1  -0.5  0  0.5  1]:\n{}'.format(g))

input('Program paused. Press ENTER to continue')

print('Initializing Neural Network Parameters ...')

initial_theta1 = rand_initialization(input_layer_size, hidden_layer_size)
initial_theta2 = rand_initialization(hidden_layer_size, num_labels)

print('Checking Backpropagation ... ')

# Check gradients by running check_nn_gradients()

lmd = 0
check_nn_gradients(lmd)

input('Program paused. Press ENTER to continue')

print('Checking Backpropagation (w/ Regularization) ...')

lmd = 3
check_nn_gradients(lmd)

# Also output the cost_function debugging values
debug_cost, _ = nn_cost_function(
    nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)

print('Cost at (fixed) debugging parameters (w/ lambda = {}): {:0.6f}\n(for lambda = 3, this value should be about 0.576051)'.format(lmd, debug_cost))

input('Program paused. Press ENTER to continue')

print('Training Neural Network ... ')

lmd = 1


def cost_func(p):
    return nn_cost_function(p, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)[0]


def grad_func(p):
    return nn_cost_function(p, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)[1]


nn_params, *unused = opt.fmin_cg(cost_func, fprime=grad_func,
                                 x0=nn_params, maxiter=400, disp=True, full_output=True)

# Obtain theta1 and theta2 back from nn_params
theta1 = nn_params[:hidden_layer_size *
                   (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
theta2 = nn_params[hidden_layer_size *
                   (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)

input('Program paused. Press ENTER to continue')


print('Visualizing Neural Network...')

display_data(theta1[:, 1:])

input('Program paused. Press ENTER to continue')


pred = predict(theta1, theta2, X)

print('Training set accuracy: {}'.format(np.mean(pred == y)*100))

input('ex4 Finished. Press ENTER to exit')
