import math

import numpy as np


class RMSProp:
    def __init__(
            self,
            learning_rate=0.001,
            beta=0.9,  # -> EMA (PAST 90% , Future 10%)
            epsilon=1e-8
    ):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.cache = None

    def setup(self, weights):
        if self.cache is None:
            self.cache = np.zeros_like(weights, dtype=float)

    def step(self, weights, gradients):
        self.setup(weights)

        self.cache = (
                self.beta * self.cache +
                (1 - self.beta) * (gradients ** 2)
        )

        effective_lr = (
                self.learning_rate /
                (np.sqrt(self.cache) + self.epsilon)
        )

        weights = (
                weights -
                effective_lr * gradients
        )

        return weights
