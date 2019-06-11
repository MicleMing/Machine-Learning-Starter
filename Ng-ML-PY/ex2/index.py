
import numpy as np
import plotData as PD

data = np.loadtxt('./data/ex2data1.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]

PD.plot_data(X, y)
