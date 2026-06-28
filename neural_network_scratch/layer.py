from neuron import Neuron
from module import Module

class Layer(Module):
    def __init__(self,neuron_count,input_size):
        self.neurons = []

        self.inputs = None
        self.outputs = None

        for _ in range(neuron_count):
            self.neurons.append(
                Neuron(input_size)
            )

    def forward(self,inputs):
        self.inputs = inputs
        outputs = []

        for neuron in self.neurons:
            outputs.append(
                neuron.forward(inputs)
            )

        self.outputs = outputs
        return outputs