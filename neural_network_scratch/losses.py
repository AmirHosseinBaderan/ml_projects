from abc import ABC, abstractmethod


class Loss(ABC):
    def __init__(self):
        self.prediction = None
        self.target = None
        self.value = None

    @abstractmethod
    def forward(self, prediction, target):
        pass

    @abstractmethod
    def derivative(self):
        pass


class MSE(Loss):

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        if len(prediction) != len(target):
            raise ValueError("prediction and target must have the same length")

        self.prediction = prediction
        self.target = target

        total = 0

        for p, t in zip(prediction, target):
            total += (p - t) ** 2

        loss = total / len(prediction)
        self.value = loss
        return loss

    def derivative(self):
        gradients = []

        n = len(self.prediction)

        for p,t in zip(self.prediction, self.target):
            gradients.append(
                (2 * (p - t) / n)
            )

        return gradients

class BinaryCrossEntropy(Loss):
    def forward(self, prediction, target):
        pass

    def derivative(self, prediction, target):
        pass
