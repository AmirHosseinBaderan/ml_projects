from collections import Counter

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sample import bootstrap_sample

class RandomForest:
    def __init__(self,n_trees=10,max_depth=None,max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.feature_subsets = []

    def fit(self,X,y):
        self.trees = []
        self.feature_subsets = []

        n_features_total = X.shape[1]
        for _ in range(self.n_trees):

            # bootstrap data
            idxs = np.random.choice(len(X),len(X),replace=True)
            X_sample = X[idxs]
            y_sample = y[idxs]

            # feature randomness
            if self.max_features is None:
                n_feats = int(np.sqrt(n_features_total))
            else:
                n_feats = self.max_features

            feat_idx = np.random.choice(n_features_total,n_feats,replace=False)
            x_subset = X_sample[:,feat_idx]

            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(x_subset,y_sample)

            self.trees.append(tree)
            self.feature_subsets.append(feat_idx)

    def predict(self,X):
        tree_preds = []

        for tree,feat_idx in zip(self.trees,self.feature_subsets):
            preds = tree.predict(X[:,feat_idx])
            tree_preds.append(preds)

        tree_preds = np.array(tree_preds)
        final_preds = []
        for i in range(X.shape[0]):
            votes = tree_preds[:,i]
            final_preds.append(Counter(votes).most_common(1)[0][0])

        return np.array(final_preds)