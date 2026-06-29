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
        if activation is None:
            activation = Identity()

        if initializer is None:
            initializer = RandomUniformInitializer()

        self.initializer = initializer
        self.activation = activation
        self.inputs = None
        self.z = None
        self.output = None
        self.weights = self.initializer.initialize_weights(input_size)
        self.bias = self.initializer.initialize_bias()
        self.delta = None
        self.weight_gradients = None
        self.bias_gradient = None

    def forward(self, inputs):
        self.inputs = inputs

        self.z = self._calculate_z(inputs)
        self.output = self.activation.activate(self.z)

        return self.output

    def backward(self,delta):
        delta *= self.activation.derivative(self.output)

        self.delta = delta

        self.weight_gradients = [
            delta * x
            for x in self.inputs
        ]

        self.bias_gradient = delta

        previous_deltas = [
            delta * weight
            for weight in self.weights
        ]

        return previous_deltas

    def _calculate_z(self, inputs):

        weighted_sum = 0

        for x, w in zip(inputs, self.weights):
            weighted_sum += x * w

        return weighted_sum + self.bias
