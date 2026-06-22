import numpy as np

def bootstrap_sample(X,y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples,n_samples,replace=True)
    return X[idxs], y[idxs]

def random_feature_subset(X,n_features):
    feature_idx = np.random.choice(X.shape[1],n_features,replace=False)
    return feature_idx