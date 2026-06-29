from module import Module
from layer import Layer


class Network(Module):

    def __init__(self):
        self.layers = []
        self.inputs = None
        self.outputs = None
        self.loss = None

    def fit(
            self,
            x_train,
            y_train,
            optimizer,
            loss,
            epochs=100,
            verbose=True,
    ):

        if len(x_train) != len(y_train):
            raise ValueError(
                "x_train and y_train must have the same length."
            )

        for epoch in range(epochs):
            epoch_loss = 0

            for inputs,target in zip(x_train, y_train):
                prediction = self.forward(inputs)

                loss_value = loss.forward(
                    prediction,
                    target
                )

                self.backward(loss)
                optimizer.step(self)
                epoch_loss += loss_value

            epoch_loss /= len(x_train)
            if verbose:
                print(
                    f"Epoch : {epoch + 1} / {epochs} | "
                    f"Loss : {epoch_loss:.6f}"
                )

    def add(self,layer):
        if not isinstance(layer, Layer):
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

    def backward(self,loss):
        deltas = loss.derivative()

        for layer in reversed(self.layers):
            deltas = layer.backward(deltas)

        self.loss = loss.value