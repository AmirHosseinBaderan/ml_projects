from abc import ABC, abstractmethod
import random


class Initializer(ABC):

    @abstractmethod
    def initialize_weights(self, input_size):
        pass

    @abstractmethod
    def initialize_bias(self):
        pass

class RandomUniformInitializer(Initializer):

    def initialize_weights(self, input_size):
        return [
            random.uniform(-1, 1)
            for _ in range(input_size)
        ]

    def initialize_bias(self):
        return random.uniform(-1, 1)

class XavierInitializer(Initializer):
    def initialize_weights(self, input_size):
        pass

    def initialize_bias(self):
        pass

class HeInitializer(Initializer):
    def initialize_weights(self, input_size):
        pass

    def initialize_bias(self):
        pass