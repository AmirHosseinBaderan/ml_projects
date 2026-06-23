from collections import Counter

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

model = RandomForest(n_trees=100,max_depth=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = np.mean(y_pred == y_test)

print(f"Accuracy : {acc}")
print(f"importance : {model.importance()}")
model.info()

print(len(model.bootstrap_indices))
print(model.bootstrap_indices[0][:20])

used = set(model.bootstrap_indices[0])

all_idx = set(range(len(X_train)))

oob = all_idx - used

print("Used:", len(used))
print("OOB:", len(oob))

sample_idx = 0
