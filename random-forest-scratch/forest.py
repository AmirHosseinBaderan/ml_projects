from collections import Counter

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sample import bootstrap_sample

class RandomForest:
    def __init__(self, n_trees=10,max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []

        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)

            x_sample,y_sample = bootstrap_sample(X,y)

            tree.fit(x_sample,y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])

        final_preds = []

        for i in range(X.shape[0]):
            votes = tree_preds[:,i]
            final_preds.append(Counter(votes).most_common(1)[0][0])

        return np.array(final_preds)