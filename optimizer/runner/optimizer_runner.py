import numpy as np

from .result import OptimizationResult


class OptimizerRunner:

    def __init__(
            self,
            optimizer,
            function):
        self.optimizer = optimizer
        self.function = function

    def run(
            self,
            start,
            iterations):
        x = np.array(start, dtype=float)

        result = OptimizationResult()

        if hasattr(self.optimizer, "velocity"):
            if self.optimizer.velocity is not None:
                result.velocity_history.append(
                    self.optimizer.velocity.copy()
                )

        for _ in range(iterations):
            result.weight_history.append(x.copy())

            result.losses_history.append(
                self.function.loss(x)
            )

            grad = self.function.gradient(x)
            result.gradients_history.append(grad.copy())
            x = self.optimizer.step(
                x,
                grad
            )

        result.weight_history.append(x.copy())
        return result
