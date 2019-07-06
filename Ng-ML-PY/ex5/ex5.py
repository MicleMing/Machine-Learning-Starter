import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

from displayData import display_data
from linearRegCostFunction import linear_reg_cost_function
from trainLinearReg import train_linear_reg
from learningCurve import learning_curve
from polyFeatures import poly_features
from featureNormalize import feature_normalize
from plotFit import plot_fit

np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

print('Loading and Visualizing data ...')

data = scio.loadmat('./data/ex5data1.mat')

X = data['X']
y = data['y'].flatten()
Xval = data['Xval']
yval = data['yval'].flatten()
Xtest = data['Xtest']
ytest = data['ytest'].flatten()

m = y.size

# display_data(X, y)

# # input('Program paused. Press ENTER to continue')

# theta = np.ones(2)
# cost, _ = linear_reg_cost_function(theta, np.c_[np.ones(m), X], y, 1)

# print(
#     'Cost at theta = [1  1]: {:0.6f}\n(this value should be about 303.993192'.format(cost))

# input('Program paused. Press ENTER to continue')

# theta = np.ones(2)
# cost, grad = linear_reg_cost_function(theta, np.c_[np.ones(m), X], y, 1)

# print(
#     'Gradient at theta = [1  1]: {}\n(this value should be about [-15.303016  598.250744]'.format(grad))

# input('Program paused. Press ENTER to continue')

# lmd = 0

# theta = train_linear_reg(np.c_[np.ones(m), X], y, lmd)

# # Plot fit over the data
# plt.plot(X, np.dot(np.c_[np.ones(m), X], theta))

# input('Program paused. Press ENTER to continue')


# lmd = 0
# error_train, error_val = learning_curve(np.c_[np.ones(m), X], y, np.c_[
#                                         np.ones(Xval.shape[0]), Xval], yval, lmd)

# plt.figure()
# plt.plot(np.arange(m), error_train, np.arange(m), error_val)
# plt.title('Learning Curve for Linear Regression')
# plt.legend(['Train', 'Cross Validation'])
# plt.xlabel('Number of Training Examples')
# plt.ylabel('Error')
# plt.axis([0, 13, 0, 150])

# input('Program paused. Press ENTER to continue')


p = 5

# Map X onto Polynomial Features and Normalize
X_poly = poly_features(X, p)
X_poly, mu, sigma = feature_normalize(X_poly)
X_poly = np.c_[np.ones(m), X_poly]

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = poly_features(Xtest, p)
X_poly_test -= mu
X_poly_test /= sigma
X_poly_test = np.c_[np.ones(X_poly_test.shape[0]), X_poly_test]

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = poly_features(Xval, p)
X_poly_val -= mu
X_poly_val /= sigma
X_poly_val = np.c_[np.ones(X_poly_val.shape[0]), X_poly_val]

print('Normalized Training Example 1 : \n{}'.format(X_poly[0]))

input('Program paused. Press ENTER to continue')

lmd = 0
theta = train_linear_reg(X_poly, y, lmd)

# Plot trainint data and fit
plt.figure()
plt.scatter(X, y, c='r', marker="x")
plot_fit(np.min(X), np.max(X), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water folowing out of the dam (y)')
plt.ylim([0, 60])
plt.title('Polynomial Regression Fit (lambda = {})'.format(lmd))

error_train, error_val = learning_curve(X_poly, y, X_poly_val, yval, lmd)
plt.figure()
plt.plot(np.arange(m), error_train, np.arange(m), error_val)
plt.title('Polynomial Regression Learning Curve (lambda = {})'.format(lmd))
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of Training Examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])

print('Polynomial Regression (lambda = {})'.format(lmd))
print('# Training Examples\tTrain Error\t\tCross Validation Error')
for i in range(m):
    print('  \t{}\t\t{}\t{}'.format(i, error_train[i], error_val[i]))

input('Program paused. Press ENTER to continue')
