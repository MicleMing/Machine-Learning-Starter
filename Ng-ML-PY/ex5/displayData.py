import matplotlib.pyplot as plt
plt.ion()


def display_data(X, y):
    plt.figure()
    plt.scatter(X, y, c='r', marker="x")
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water folowing out of the dam (y)')
