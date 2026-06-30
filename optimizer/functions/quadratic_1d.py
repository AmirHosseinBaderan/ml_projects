from .base import Function

class Quadratic1D(Function):

    def loss(self, x):
        return x ** 2

    def gradient(self, x):
        return 2 * x