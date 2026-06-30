import matplotlib.pyplot as plt

class Visualizer:
    def plot_loss(self, result):
        plt.figure(figsize=(8, 5))

        plt.plot(
            result.losses_history,
            linewidth=2,
        )

        plt.title("Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")

        plt.grid(True)
        plt.show()

    def plot_weight(self,result):

        plt.figure(figsize=(8, 5))

        weights = [
            w[0]
            for w in result.weight_history
        ]

        plt.plot(weights)
        plt.title("Weights")

        plt.grid(True)
        plt.show()
