import random

import numpy as np


class Dropout:
    def __init__(
            self,
            p=0.5
    ):
        if not 0 <= p < 1:
            raise ValueError(
                "Dropout probability must be in the range [0, 1)."
            )
        self.p = p
        self.mask = None

    def forward(self, x, training=True):
        if not training:
            return x

        self.mask = (
                np.random.rand(*x.shape) >= self.p
        ).astype(float)

        output = x * self.mask
        output /= (1 - self.p)
        return output
