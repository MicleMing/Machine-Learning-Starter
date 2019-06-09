import numpy as np

arr1 = np.arange(10)

arr2 = np.full((3, 3), True, dtype=bool)

arr3 = np.ones((3, 3), dtype=bool)

arr4 = np.array([0, 1, 2, 3, 4, 5])


def replace(num):
    arr = np.arange(10)
    out = np.where(arr % 2 == 1, -1, arr)
    print(out)


def reshape():
    arr = np.arange(10)
    out = arr.reshape(2, -1)
    print(out)


def v_concat():
    arr1 = np.arange(10).reshape(2, -1)
    arr2 = np.repeat(1, 10).reshape(2, -1)
    out = np.r_[arr1, arr2]
    print(out)


def h_concat():
    arr1 = np.arange(10).reshape(2, -1)
    arr2 = np.repeat(1, 10).reshape(2, -1)
    out = np.c_[arr1, arr2]
    print(out)


def add():
    arr = np.array([1, 2, 3])
    r = np.repeat(arr, 3)
    t = np.tile(arr, 3)
    out = np.r_[r, t]
    print(out)
