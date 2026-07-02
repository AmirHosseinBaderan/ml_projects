import numpy as np

from .base import Regularizer


class L2(Regularizer):

    def __init__(self, lambda_=0.01):
        self.lambda_ = lambda_

    def penalty(self, weights):
        # weights_power = [
        #     w ** 2
        #     for w in weights
        # ]
        # return self.lambda_ * np.sum(weights_power)
        return self.lambda_ * np.sum(np.square(weights))

    def gradient(self, weights):
        return 2 * self.lambda_ * weights
