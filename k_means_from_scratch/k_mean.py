import math
import random


def euclidean_distance(p1, p2):
    total = 0

    for a, b in zip(p1, p2):
        total += (a - b) ** 2

    return math.sqrt(total)


class KMeans:
    def __init__(self, k=2, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = []

    def init_centroids(self, X):
        self.centroids = random.sample(X, self.k)

    def assign_clusters(self, X):
        clusters = [[] for _ in range(self.k)]

        for point in X:
            distances = []
            for centroid in self.centroids:
                dist = euclidean_distance(
                    point,
                    centroid
                )

                distances.append(dist)

            closest_idx = distances.index(
                min(distances)
            )
            clusters[closest_idx].append(point)

        return clusters

    def update_centroids(self, clusters):
        centroids = []
        for cluster in clusters:
            centroid = []

            for feature_value in zip(*cluster):
                mean = sum(feature_value) / len(feature_value)
                centroid.append(mean)
            centroids.append(centroid)
        self.centroids = centroids

    def fit(self,X):
        self.init_centroids(X)

        for _ in range(self.max_iters):
            old_centroids = self.centroids.copy()

            clusters = self.assign_clusters(X)
            self.update_centroids(clusters)

            if old_centroids == self.centroids:
                break

    def predict(self,point):
        distances = []

        for centroid in self.centroids:
            dist = euclidean_distance(
                point,
                centroid
            )

            distances.append(dist)

        return distances.index(
            min(distances)
        )