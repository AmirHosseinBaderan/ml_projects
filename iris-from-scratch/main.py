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

print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)