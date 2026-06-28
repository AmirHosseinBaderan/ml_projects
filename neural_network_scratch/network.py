from module import Module

class Network(Module):

    def __init__(self):
        self.layers = []
        self.inputs = None
        self.outputs = None

    def add(self,layer):
        if not isinstance(layer, Module):
            raise TypeError("layer must be an instance of Layer")

        self.layers.append(layer)

    def forward(self,inputs):
        self.inputs = inputs
        outputs = inputs

        for layer in self.layers:
            outputs = layer.forward(outputs)

        self.outputs = outputs
        return outputs

    def predict(self,inputs):
        return self.forward(inputs)