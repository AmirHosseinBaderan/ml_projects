import matplotlib.pyplot as plt
import numpy as np


class Function1DVisualizer:

    def plot(self,
             function,
             result,
             x_min=-10,
             x_max=10):
        x = np.linspace(
            x_min,
            x_max,
            500
        )

        y = function.loss(x)
        plt.figure(figsize=(10, 6))

        plt.plot(x, y)

        weights = np.array(
            result.weight_history
        ).flatten()
        losses = function.loss(weights)

        plt.scatter(
            weights,
            losses,
            color="red",
        )
        plt.plot(
            weights,
            losses,
            "--",
            color="red",
            alpha=0.5
        )

        for i in range(len(weights)):
            plt.text(
                weights[i],
                losses[i],
                str(i),
            )

        plt.title("Optimizer Path")

        plt.xlabel("Weight")

        plt.ylabel("Loss")

        plt.grid(True)

        plt.show()
