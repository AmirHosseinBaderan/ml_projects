from neuron import Neuron
from module import Module


class Layer(Module):
    def __init__(
            self,
            neuron_count,
            input_size,
            activation=None,
            initializer=None,
    ):
        self.neurons = []

        self.inputs = None
        self.outputs = None
        self.input_size = input_size
        self.deltas = None
        self.activation = activation
        self.initializer = initializer

        for _ in range(neuron_count):
            self.neurons.append(
                Neuron(
                    input_size,
                    activation=activation,
                    initializer=initializer,
                )
            )

    def forward(self, inputs):
        self.inputs = inputs
        outputs = []

        for neuron in self.neurons:
            outputs.append(
                neuron.forward(inputs)
            )

        self.outputs = outputs
        return outputs

    def backward(self, deltas):
        if len(deltas) != len(self.neurons):
            raise ValueError(
                "Number of deltas must match number of neurons."
            )

        accumulated = [0] * self.input_size

        for neuron, delta in zip(self.neurons, deltas):
            previous = neuron.backward(delta)

            for i, value in enumerate(previous):
                accumulated[i] += value

        self.deltas = accumulated
        return accumulated
