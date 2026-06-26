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
        self.visited = []

    def region_query(self,X,point_index):
        point = X[point_index]
        neighbors = []

        for i,other_point in enumerate(X):
            distance = euclidean_distance(point,other_point)
            if distance <= self.eps:
                neighbors.append(i)

        return neighbors

    def expand_cluster(self, X, point_index, neighbors, cluster_id):

        self.labels[point_index] = cluster_id

        i = 0

        while i < len(neighbors):

            neighbor = neighbors[i]

            if not self.visited[neighbor]:

                self.visited[neighbor] = True

                neighbor_neighbors = self.region_query(X, neighbor)

                if len(neighbor_neighbors) >= self.min_samples:

                    for new_neighbor in neighbor_neighbors:
                        if new_neighbor not in neighbors:
                            neighbors.append(new_neighbor)

            if self.labels[neighbor] == -1:
                self.labels[neighbor] = cluster_id

            i += 1

    def fit(self, X):

        self.labels = [-1] * len(X)
        self.visited = [False] * len(X)

        cluster_id = 0

        for point_index in range(len(X)):

            if self.visited[point_index]:
                continue

            self.visited[point_index] = True

            neighbors = self.region_query(X, point_index)

            if len(neighbors) < self.min_samples:
                continue

            self.expand_cluster(
                X,
                point_index,
                neighbors,
                cluster_id
            )

            cluster_id += 1

    def fit_predict(self,X):
        self.fit(X)
        return self.labels

