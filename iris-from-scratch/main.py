import math
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/iris.csv")

# Features , Label
X = df.drop(columns=["variety"])
y = df["variety"]

# train test split
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# distance
def euclidean_distance(x, y):

    n = len(x)
    sum_squared = 0
    for i in range(n):
        sum_squared += (x[i] - y[i]) ** 2

    return math.sqrt(sum_squared)

a = [5.1, 3.5, 1.4, 0.2]
b = [5.2, 3.4, 1.5, 0.3]
c = [7.0, 3.2, 5.9, 2.1]
print(euclidean_distance(a,b))
print(euclidean_distance(a, c))
