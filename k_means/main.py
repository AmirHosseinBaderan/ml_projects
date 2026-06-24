from sklearn.cluster import KMeans
import pandas as pd

customers = {
    "orders":[
        1,2,3,5,4,
        80,90,100,120,95
    ],
    "spend":[
        50,100,120,200,180,
        7000,8500,9000,11000,9200
    ]
}
df = pd.DataFrame(customers)

model = KMeans(
    n_clusters=3,
    random_state=42
)

clusters = model.fit_predict(df)

print(clusters)
print(model.cluster_centers_)