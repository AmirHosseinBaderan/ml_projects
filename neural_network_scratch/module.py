from abc import ABC, abstractmethod

class Module(ABC):
    @abstractmethod
    def forward(self, inputs):
        pass