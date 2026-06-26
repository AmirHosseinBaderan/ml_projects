import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

X = [
    [1, 2],
    [2, 2],
    [2, 3],
    [8, 8],
    [8, 9],
    [25, 25]
]

x = [p[0] for p in X]
y = [p[1] for p in X]

plt.scatter(x,y)
plt.grid(True)
plt.show()

model = DBSCAN(
    eps=20,
    min_samples=2,
)

labels = model.fit_predict(X)
print(labels)

# show result
plt.scatter(
    x,
    y,
    c=labels,
    s=100,
)

plt.grid(True)
plt.show()