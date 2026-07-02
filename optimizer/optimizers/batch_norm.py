import numpy as np


class BatchNorm:
    def __init__(
            self,
            num_features,
            momentum=0.9,
            epsilon=1e-5,
    ):
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.momentum = momentum
        self.epsilon = epsilon

    def forward(
            self,
            X,
            training=True,
    ):
        if training:

            mu = np.mean(X, axis=0)
            var = np.var(X, axis=0)

            self.running_mean = (
                    self.momentum * self.running_mean
                    +
                    (1 - self.momentum) * mu
            )

            self.running_var = (
                    self.momentum * self.running_var
                    +
                    (1 - self.momentum) * var
            )

        else:

            mu = self.running_mean
            var = self.running_var

        x_hat = (
                        X - mu
                ) / np.sqrt(
            var + self.epsilon
        )

        Y = self.gamma * x_hat + self.beta

        return Y