import math

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

    def fit(self, x_train, y_train):
        self.x_train = x_train.to_numpy()
        self.y_train = y_train.to_numpy()

    def predict(self, sample):
        items = []
        for i in range(len(self.x_train)):
            x_sample = self.x_train[i]
            y_sample = self.y_train[i]
            distance = euclidean_distance(x_sample, sample)
            items.append((distance, y_sample))

        items.sort(key=lambda x: x[0])
        neighbors = items[:self.k]

        votes = {}
        for _, label in neighbors:
            votes[label] = votes.get(label, 0) + 1

        return max(votes, key=votes.get)

    def predict_many(self,samples):
        predictions = []
        samples_numpy = samples.to_numpy()
        for i in range(len(samples)):
            predictions.append(self.predict(samples_numpy[i]))
        return predictions