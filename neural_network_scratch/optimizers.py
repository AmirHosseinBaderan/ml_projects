from abc import ABC, abstractmethod


class Optimizer(ABC):

    @abstractmethod
    def step(self, network):
        pass

class SGD(Optimizer):

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, network):
        pass