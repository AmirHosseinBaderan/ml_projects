from abc import ABC,abstractmethod
import math

class Activation(ABC):
    @abstractmethod
    def activate(self,x):
        pass

    @abstractmethod
    def derivative(self, output):
        pass

class Identity(Activation):

    def activate(self,x):
        return x

    def derivative(self, output):
        return 1

class Sigmoid(Activation):
    def activate(self,x):
        return 1 / (1 + math.exp(-x))

    def derivative(self, output):
        return output * (1 - output)


class Tanh(Activation):
    def activate(self,x):
        # numerator = math.exp(x) - math.exp(-x)
        # denominator = math.exp(x) + math.exp(-x)
        # return numerator / denominator
        return math.tanh(x)

    def derivative(self, output):
        return 1 - output ** 2