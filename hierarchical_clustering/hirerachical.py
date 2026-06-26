import math

def euclidean_distance(p1, p2):
    total = 0

    for a, b in zip(p1, p2):
        total += (a - b) ** 2

    return math.sqrt(total)

class HierarchicalClustering:
    def __init__(self):
        self.clusters = []

    def cluster_distance(self, cluster1, cluster2):
        min_distance = float("inf")

        for p1 in cluster1:
            for p2 in cluster2:
                distance = euclidean_distance(p1, p2)

                if distance < min_distance:
                    min_distance = distance

        return min_distance

    def find_closest_cluster(self):
        min_distance = float("inf")

        best_i = None
        best_j = None

        for i in range(len(self.clusters)):
            curr_cluster = self.clusters[i]

            for j in range(i+1, len(self.clusters)):
                next_cluster = self.clusters[j]

                distance = self.cluster_distance(curr_cluster, next_cluster)
                if distance < min_distance:
                    min_distance = distance
                    best_i = i
                    best_j = j

        return best_i, best_j,min_distance

    def merge_clusters(self, i, j):
        self.clusters[i].extend(self.clusters[j])
        del self.clusters[j]

    def fit(self,X,n_clusters=1):
        self.clusters = []

        for point in X:
            self.clusters.append([point])

        while len(self.clusters) > n_clusters:
            i,j,distance = self.find_closest_cluster()

            self.merge_clusters(i,j)

        return self.clusters