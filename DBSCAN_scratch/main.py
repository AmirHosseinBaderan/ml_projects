from DBSCAN import DBSCAN

X = [
    [1, 2],
    [2, 2],
    [2, 3],
    [8, 8],
    [8, 9],
    [25, 25]
]

db = DBSCAN(eps=2,min_samples=2)

print(db.fit_predict(X))