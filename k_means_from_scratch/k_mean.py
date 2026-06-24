import math
import random

def euclidean_distance(p1, p2):
    total = 0

    for a, b in zip(p1, p2):
        total += (a - b) ** 2

    return math.sqrt(total)

class KMeans:
    def __init__(self,k=2,max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = []

    def init_centroids(self,X):
        self.centroids = random.sample(X,self.k)

