import numpy as np

def bootstrap_sample(X,y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples,n_samples,replace=True)
    return X[idxs], y[idxs]
