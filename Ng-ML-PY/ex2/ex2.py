
import numpy as np
import costFunction as Cost
import scipy.optimize as opt
from sigmoid import *
from predict import *
from plotData import *
from plotDecisionBoundary import *

data = np.loadtxt('./data/ex2data1.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]

plot_data(X, y)


(m, n) = X.shape

# Add intercept term
X = np.c_[np.ones(m), X]

# Initialize fitting parameters
initial_theta = np.zeros(n + 1)

cost, grad = Cost.cost_function(initial_theta, X, y)

np.set_printoptions(formatter={'float': '{: 0.3f}\n'.format})

print('Cost at initial theta (zeros): {:0.3f}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros): \n{}'.format(grad))
print('Expected gradients (approx): \n-0.1000\n-12.0092\n-11.2628')

test_theta = np.array([-24, 0.2, 0.2])
cost, grad = Cost.cost_function(test_theta, X, y)

print('Cost at test theta (zeros): {:0.3f}'.format(cost))
print('Expected cost (approx): 0.218')
print('Gradient at test theta: \n{}'.format(grad))
print('Expected gradients (approx): \n0.043\n2.566\n2.647')


def cost_func(t):
    return Cost.cost_function(t, X, y)[0]


def grad_func(t):
    return Cost.cost_function(t, X, y)[1]


# Run fmin_bfgs to obtain the optimal theta
theta, cost, *unused = opt.fmin_bfgs(f=cost_func, fprime=grad_func,
                                     x0=initial_theta, maxiter=400, full_output=True, disp=False)

print('Cost at theta found by fmin: {:0.4f}'.format(cost))
print('Expected cost (approx): 0.203')
print('theta: \n{}'.format(theta))
print('Expected Theta (approx): \n-25.161\n0.206\n0.201')

plot_decision_boundary(theta, X, y)

scores = np.array([1, 45, 85])

prob = sigmoid(scores.dot(theta))
print(
    'For a student with scores 45 and 85, we predict an admission probability of {:0.4f}'.format(prob))
print('Expected value : 0.775 +/- 0.002')

p = predict(theta, X)

print('Train accuracy: {}'.format(np.mean(y == p) * 100))
print('Expected accuracy (approx): 89.0')

input('ex2 Finished. Press ENTER to exit')
