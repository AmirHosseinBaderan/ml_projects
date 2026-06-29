from abc import ABC, abstractmethod


class Initializer(ABC):
    @abstractmethod
    def initialize(self,input_size):
        pass

class RandomUniformInitializer(Initializer):
    def initialize(self,input_size):
        pass

class XavierInitializer(Initializer):
    def initialize(self,input_size):
        pass

class HeInitializer(Initializer):
    def initialize(self,input_size):
        pass