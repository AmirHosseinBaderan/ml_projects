from abc import ABC, abstractmethod


class Regularizer(ABC):

    @abstractmethod
    def penalty(self, weights):
        pass

    @abstractmethod
    def gradient(self, weights):
        pass