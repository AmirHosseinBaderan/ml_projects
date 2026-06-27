from isolation_forest import IsolationForest
import numpy as np

X = np.array([
    [10],
    [12],
    [13],
    [15],
    [14],
    [11],
    [90],
    [95]
])

model = IsolationForest(
    n_trees=100,
    max_samples=8
)

model.fit(X)

result = model.predict(X)

for sample, score, label in result:
    print(
        f"{sample[0]:>3} | score = {score:.3f} | {label}"
    )


normal = np.random.normal(
    loc=50,
    scale=5,
    size=(100, 2)
)

anomaly = np.array([
    [120, 130],
    [150, 160],
    [200, 180]
])

X = np.vstack((normal, anomaly))

model = IsolationForest(
    n_trees=100,
    max_samples=64
)

model.fit(X)

result = model.predict(X)

for sample, score, label in result:
    if label == "anomaly":
        print(
            f"{sample[0]:>3} | score = {score:.3f} | {label}"
        )