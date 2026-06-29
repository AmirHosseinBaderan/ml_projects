from abc import ABC, abstractmethod


class Optimizer(ABC):

    @abstractmethod
    def step(self, network):
        pass

class SGD(Optimizer):

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, network):

        for layer in network.layers:

            for neuron in layer.neurons:

                if neuron.weight_gradients is None:
                    raise RuntimeError(
                        "Call backward() before optimizer.step()."
                    )

                for i, gradient in enumerate(neuron.weight_gradients):
                    neuron.weights[i] -= (
                        self.learning_rate * gradient
                    )

                neuron.bias -= (
                    self.learning_rate * neuron.bias_gradient
                )