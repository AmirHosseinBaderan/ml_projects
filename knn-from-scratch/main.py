import pandas as pd
from knn import euclidean_distance,get_neighbors,predict

df = pd.read_csv('./data/iris.csv')

# X
distance_df = df.drop(columns=['variety'])
y = df['variety']

sample = distance_df.loc[1].values
neighbors = get_neighbors(distance_df,y,sample,k=3)

print(neighbors)

print(euclidean_distance(distance_df.loc[1].values,distance_df.loc[2].values))

result = predict(distance_df,y,sample,k=3)
print(result)

current = 0
for i in range(len(distance_df)):
    pred = predict(distance_df,y,distance_df.iloc[i].values,k=3)
    if pred == y.iloc[i]:
        current += 1
        
print(f" Accuracy : {current / len(distance_df)}")