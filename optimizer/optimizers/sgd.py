class SGD:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def step(self, weights, gradients):
        return weights - self.learning_rate * gradients
