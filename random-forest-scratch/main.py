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

n_features = X_train.shape[1]

forest_importance = np.zeros(n_features)

for tree, feature_idx in zip(
    model.trees,
    model.feature_subsets
):
    for idx, importance in zip(
        feature_idx,
        tree.feature_importances_
    ):
        forest_importance[idx] += importance

forest_importance /= len(model.trees)
print(forest_importance)
feature_names = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

for name, score in zip(
    feature_names,
    forest_importance
):
    print(
        f"{name}: {score:.4f}"
    )