import numpy as np
from pca import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = np.array([
    [170,65],
    [172,67],
    [175,70],
    [180,75],
    [183,78]
])

X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=1)

X_new = pca.fit_transform(X_scaled)

print(X_new)
print(pca.explained_variance_ratio_)

X_original = pca.inverse_transform(X_new)

print(X_original)