from abc import ABC, abstractmethod


class Initializer(ABC):

    @abstractmethod
    def initialize_weights(self, input_size):
        pass

    @abstractmethod
    def initialize_bias(self):
        pass

class RandomUniformInitializer(Initializer):
    def initialize_weights(self,input_size):
        pass

    def initialize_bias(self):
        pass

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