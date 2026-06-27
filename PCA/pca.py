import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.array(X, dtype=float)

        # mean clustering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # covariance matrix
        cov = np.cov(X_centered.T)

        # Eigen Decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort Eigenvalues
        indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]

        total = np.sum(eigenvalues)

        self.explained_variance_ratio_ = (
                eigenvalues[:self.n_components] / total
        )

        # Keep Top Components
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        X = np.array(X, dtype=float)
        X_centered = X - self.mean

        X_new = X_centered @ self.components

        return X_new

    def fit_transform(self, X):
        self.fit(X)

        return self.transform(X)

    def inverse_transform(self, X):
        X = np.array(X, dtype=float)

        return (X @ self.components.T) + self.mean
