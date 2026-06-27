import numpy as np
from sklearn.ensemble import IsolationForest

X = np.array([
    [10],
    [12],
    [13],
    [11],
    [15],
    [14],
    [90],
    [95]
])

model = IsolationForest(
    n_estimators=100,
    max_samples='auto',
    contamination=0.25,
    random_state=42
)

model.fit(X)

predictions = model.predict(X)

print(predictions)

scores = model.score_samples(X)

print(scores)

print(model.offset_)