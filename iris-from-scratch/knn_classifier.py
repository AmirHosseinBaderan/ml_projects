import math
import operator


def euclidean_distance(x, y):
    n = len(x)
    sum_squared = 0
    for i in range(n):
        sum_squared += (x[i] - y[i]) ** 2

    return math.sqrt(sum_squared)

class KNNClassifier:
    def __init__(self, k=3):
        self.y_train = None
        self.x_train = None
        self.k = k

    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self,sample):
        items = []
        for i in range(len(self.x_train)):
            x_sample = self.x_train[i]
            y_sample = self.y_train[i]
            distance = euclidean_distance(x_sample, sample)
            items.append((distance,y_sample))

        items.sort(key=lambda x: x[0])
        neighbors = items[:self.k]

        votes = {}
        for _,label in neighbors:
            if votes.get(label) is None:
                votes[label] = 1
            else:
                votes[label] += 1

        predict = ''
        largest = 0
        for key in votes.keys():
            value = votes[key]
            if value > largest:
                largest = value
                predict = key

        return predict