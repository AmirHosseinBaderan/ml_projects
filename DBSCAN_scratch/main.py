from DBSCAN import DBSCAN

X = [
    [1],
    [2],
    [3],
    [10]
]

db = DBSCAN(eps=1.5)

print(db.region_query(X,1))