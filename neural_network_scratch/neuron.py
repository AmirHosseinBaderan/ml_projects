import random
from module import Module
from activations import Identity


class Neuron(Module):
    def __init__(
            self,
            input_size,
            activation=Identity()
    ):
        self.weights = [
            random.uniform(-1,1)
            for _ in range(input_size)
        ]
        self.bias = random.uniform(-1,1)
        self.activation = activation

        self.inputs = None
        self.z = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs

        self.z = self._calculate_z(inputs)
        self.output = self.activation.activate(self.z)

        return self.output

    def _calculate_z(self, inputs):

        weighted_sum = 0

        for x, w in zip(inputs, self.weights):
            weighted_sum += x * w

        return weighted_sum + self.bias

