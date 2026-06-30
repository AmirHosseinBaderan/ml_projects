from abc import ABC, abstractmethod


class Function(ABC):

    @abstractmethod
    def loss(self, x):
        pass

    @abstractmethod
    def gradient(self, x):
        pass