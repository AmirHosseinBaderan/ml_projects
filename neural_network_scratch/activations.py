from abc import ABC,abstractmethod


class Activation(ABC):
    @abstractmethod
    def activate(self,x):
        pass

class Identity(Activation):

    def activate(self,x):
        return x