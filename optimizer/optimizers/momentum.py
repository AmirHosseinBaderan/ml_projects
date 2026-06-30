import numpy as np

class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def step(self, weights, gradients):
        if self.velocity is None:
            self.velocity = np.zeros_like(weights)

        self.velocity = (
                self.momentum * self.velocity
                - self.learning_rate * gradients
        )

        weights += self.velocity

        return weights
