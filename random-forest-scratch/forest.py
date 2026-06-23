from collections import Counter

import numpy as np
from sklearn.tree import DecisionTreeClassifier

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.feature_subsets = []
        self.bootstrap_indices = []

    def fit(self, X, y):
        self.trees = []
        self.feature_subsets = []

        n_features_total = X.shape[1]
        for _ in range(self.n_trees):

            # bootstrap data
            idxs = np.random.choice(len(X), len(X), replace=True)
            X_sample = X[idxs]
            y_sample = y[idxs]

            self.bootstrap_indices.append(idxs)

            # feature randomness
            if self.max_features is None:
                n_feats = int(np.sqrt(n_features_total))
            else:
                n_feats = self.max_features

            feat_idx = np.random.choice(n_features_total, n_feats, replace=False)
            x_subset = X_sample[:, feat_idx]

            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(x_subset, y_sample)

            self.trees.append(tree)
            self.feature_subsets.append(feat_idx)

    def predict(self, X):
        tree_preds = []

        for tree, feat_idx in zip(self.trees, self.feature_subsets):
            preds = tree.predict(X[:, feat_idx])
            tree_preds.append(preds)

        tree_preds = np.array(tree_preds)
        final_preds = []
        for i in range(X.shape[0]):
            votes = tree_preds[:, i]
            final_preds.append(Counter(votes).most_common(1)[0][0])

        return np.array(final_preds)

    def importance(self):
        all_importances = []

        for tree in self.trees:
            all_importances.append(tree.feature_importances_)

        res = np.mean(
            all_importances,
            axis=0
        )

        return res

    def info(self):
        print(len(self.trees))
        print(self.feature_subsets[0])
        print(
            self.trees[0].feature_importances_
        )

    def oob_predict(self, X, sample_idx):
        votes = []

        for tree_idx, bootstrap_idx in enumerate(self.bootstrap_indices):
            if sample_idx not in bootstrap_idx:
                tree = self.trees[tree_idx]
                feat_idx = self.feature_subsets[tree_idx]

                x = X[sample_idx].reshape(1, -1)[:, feat_idx]

                pred = tree.predict(x)[0]

                votes.append(pred)

        if len(votes) == 0:
            return None

        return Counter(votes).most_common(1)[0][0]

    def oob_score(self, X, y):
        correct = 0
        total = 0

        n_samples = X.shape[0]

        for sample_idx in range(n_samples):
            votes = []

            for tree_idx, bootstrap_idx in enumerate(self.bootstrap_indices):

                if sample_idx not in bootstrap_idx:
                    tree = self.trees[tree_idx]
                    feat_idx = self.feature_subsets[tree_idx]

                    x = X[sample_idx].reshape(1, -1)[:, feat_idx]

                    pred = tree.predict(x)[0]
                    votes.append(pred)

            if len(votes) == 0:
                continue

            final_pred = Counter(votes).most_common(1)[0][0]

            if final_pred == y[sample_idx]:
                correct += 1

            total += 1

        return correct / total
