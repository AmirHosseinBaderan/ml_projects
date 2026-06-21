import math
import pandas as pd
from sklearn.model_selection import train_test_split
from knn_classifier import KNNClassifier

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

model = KNNClassifier(3)
model.fit(X_train, y_train)

# Accuracy
x_pred = model.predict_many(X_test)
def accuracy(y_true, y_pred):
    correct = 0
    y_true = y_true.to_numpy()
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            correct += 1

    return correct / len(y_true)

acc = accuracy(y_test, x_pred)
print(acc)