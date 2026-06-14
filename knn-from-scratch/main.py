import pandas as pd
from knn import euclidean_distance,get_neighbors,predict

df = pd.read_csv('./data/iris.csv')

# X
X = df.drop(columns=['variety'])
y = df['variety']

sample = X.iloc[1].values
neighbors = get_neighbors(X,y,sample,k=3)

print(neighbors)

print(euclidean_distance(X.loc[1].values,X.loc[2].values))

result = predict(X,y,sample,k=3)
print(result)

current = 0
for i in range(len(X)):
    pred = predict(X,y,X.iloc[i].values,k=3)
    if pred == y.iloc[i]:
        current += 1
        
print(f" Accuracy : {current / len(X)}")