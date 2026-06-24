import math

from k_mean import euclidean_distance, KMeans

X = [
    [1, 50], [2, 100], [3, 120], [5, 200], [4, 180],
    [80, 7000], [90, 8500], [100, 9000], [120, 11000], [95, 9200]
]

model = KMeans()
model.fit(X)

print(model.centroids)
print(model.assign_clusters(X))

print(
    f"predict one : {model.predict(
        [95, 9000]
    )}"
)
print(
    f"predict two : {model.predict(
        [2, 120]
    )}"
)
