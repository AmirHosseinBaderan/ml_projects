import math
import random

from node import Node


class IsolationForest:
    def __init__(
            self,
            n_trees=100,
            max_samples=250,
            max_depth=None
    ):
        self.n_trees = n_trees
        self.max_samples = max_samples
        self.max_depth = max_depth

        self.trees = []

    def fit(self, X):
        self.trees = []

        if self.max_depth is None:
            self.max_depth = math.ceil(math.log2(self.max_samples))

        n_samples = min(len(X),self.max_samples)

        for _ in range(self.n_trees):
            indices = random.sample(
                range(len(X)),
                n_samples
            )

            subset = X[indices]
            tree = self._build_tree(
                subset,
                depth=0
            )

            self.trees.append(tree)

    def predict(self, X):
        result = []
        for sample in X:
            paths = []

            for tree in self.trees:
                paths.append(
                    self._path_length(
                        sample,
                        tree,
                        0
                    )
                )

            avg = sum(paths) / len(paths)
            score = 2 ** (
                -avg / self._c(self.max_samples)
            )

            if score > 0.7:
                result.append((sample, score,"anomaly"))
            else:
                result.append((sample, score,"normal"))

        return result

    def _path_length(
            self,
            x,
            node,
            depth
    ):
        if node.left is None and node.right is None:
            return depth + self._c(node.size)

        if x[node.feature] < node.split_value:
            return self._path_length(
                x,
                node.left,
                depth + 1
            )

        return self._path_length(
            x,
            node.right,
            depth + 1
        )

    def _build_tree(
            self,
            X,
            depth
    ):
        if len(X) <= self.max_depth or depth >= self.max_depth:
            return Node(size=len(X))

        feature = random.randint(
            0,
            X.shape[1] - 1
        )

        values = X[:, feature]
        min_val = values.min()
        max_val = values.max()

        if min_val == max_val:
            return Node(size=len(X))

        split = random.uniform(
            min_val,
            max_val
        )
        left_mask = values < split
        right_mask = ~left_mask

        left_X = X[left_mask]
        right_X = X[right_mask]

        left = self._build_tree(
            left_X,
            depth + 1
        )

        right = self._build_tree(
            right_X,
            depth + 1
        )

        return Node(
            feature=feature,
            split_value=split,
            left=left,
            right=right
        )

    def _c(self, n):
        if n <= 1:
            return 0

        return (
                2 * (math.log(n - 1) + 0.5772156649)
                - (2 * (n - 1) / n)
        )
