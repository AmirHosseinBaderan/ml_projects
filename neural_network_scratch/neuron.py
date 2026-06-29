import random
from module import Module
from activations import Identity
from initializer import  RandomUniformInitializer


class Neuron(Module):
    def __init__(
            self,
            input_size,
            activation=None,
            initializer=None,
    ):
        self.bias = random.uniform(-1, 1)
        if activation is None:
            activation = Identity()

        if initializer is None:
            initializer = RandomUniformInitializer()

        self.initializer = initializer
        self.activation = activation
        self.inputs = None
        self.z = None
        self.output = None
        self.weights = self.initializer.initialize(input_size)

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
