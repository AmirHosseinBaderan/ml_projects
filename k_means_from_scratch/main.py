import math
from k_mean import euclidean_distance, KMeans

X = [
    [1,50],
    [2,100],
    [3,120],
    [120,11000]
]

model = KMeans()
model.init_centroids(X)

print(model.centroids)