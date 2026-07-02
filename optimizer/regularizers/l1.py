import numpy as np

from .base import Regularizer


class L1(Regularizer):
    def __init__(self, lambda_=0.01):
        self.lambda_ = lambda_

    def penalty(self, weights):
        abs_weights = np.abs(weights)
        abs_sum = np.sum(abs_weights)

        return self.lambda_ * abs_sum

    def gradient(self, weights):
        return self.lambda_ * np.sign(weights)
