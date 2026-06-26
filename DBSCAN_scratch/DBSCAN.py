import math

def euclidean_distance(p1, p2):
    total = 0

    for a, b in zip(p1, p2):
        total += (a - b) ** 2

    return math.sqrt(total)


class DBSCAN:
    def __init__(self,eps=0.5,min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = []

    def region_query(self,X,point_index):
        point = X[point_index]
        neighbors = []

        for i,other_point in enumerate(X):
            distance = euclidean_distance(point,other_point)
            if distance <= self.eps:
                neighbors.append(i)

        return neighbors


    def expand_cluster(self, X, point_index, neighbors, cluster_id):
        pass

    def fit(self,X):
        pass

    def fit_predict(self,X):
        pass

