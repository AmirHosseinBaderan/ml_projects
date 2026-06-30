import numpy as np
import matplotlib.pyplot as plt

class ContourVisualizer:

    def plot(self, function, result):
        x = np.linspace(-10, 10, 200)
        y = np.linspace(-10, 10, 200)

        X, Y = np.meshgrid(x, y)
        Z = function.loss(np.array([X, Y]))

        plt.contour(
            X,
            Y,
            Z,
            levels=20
        )

        weights = np.array(result.weight_history)

        xs = weights[:, 0]
        ys = weights[:, 1]

        plt.plot(
            xs,
            ys,
            color="red",
            marker="o"
        )

        for i in range(len(xs)):
            plt.text(
                xs[i],
                ys[i],
                str(i)
            )

        plt.xlabel("x")
        plt.ylabel("y")

        plt.title("Optimizer Path")

        plt.grid(True)

        plt.show()