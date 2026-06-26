import math
from hirerachical import HierarchicalClustering

model = HierarchicalClustering()

X = [
    [1],
    [2],
    [10],
    [11]
]

fit = model.fit(X, n_clusters=2)
print(fit)