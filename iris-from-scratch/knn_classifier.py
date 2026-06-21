import math
import operator


def euclidean_distance(x, y):
    n = len(x)
    sum_squared = 0
    for i in range(n):
        sum_squared += (x[i] - y[i]) ** 2

    return math.sqrt(sum_squared)

