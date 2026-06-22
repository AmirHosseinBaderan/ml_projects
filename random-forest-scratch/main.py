import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from forest import RandomForest

df = pd.read_csv("./data/iris.csv")
X = df.drop(columns=['variety']).values
y = df['variety'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

model = RandomForest(n_trees=100,max_depth=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = np.mean(y_pred == y_test)

print(f"Accuracy : {acc}")