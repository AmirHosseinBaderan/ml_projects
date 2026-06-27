import numpy as np
from sklearn.decomposition import PCA

X = np.array([
    [170,65],
    [172,67],
    [175,70],
    [180,75],
    [183,78]
])

model = PCA(n_components=2)
X_new = model.fit_transform(X)
print(X_new)