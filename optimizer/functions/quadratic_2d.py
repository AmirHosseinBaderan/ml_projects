import numpy as np

from .base import Function

class Quadratic2D(Function):

    def loss(self, x):
        x_x = x[0]
        x_y = x[1]

        return x_x ** 2 + 10 * x_y ** 2

    def gradient(self, x):
        return np.array([
            2 * x[0],
            20 * x[1]
        ])