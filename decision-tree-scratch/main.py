import operator
from sklearn.model_selection 

class DecisionNode:
    def __init__(self,feature_idx,threshold,left,right):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right

class LeafNode:
    def __init__(self,value):
        self.value = value


def gini(y):
    counts = {}

    for label in y:
        counts[label] = counts.get(label, 0) + 1

    impurity = 1
    total = len(y)

    for label in counts:
        p = counts[label] / total
        impurity -= p ** 2

    return impurity

def get_threshold(feature_column):
    sorted_vals = sorted(set(feature_column))

    threshold = []

    for i in range(len(sorted_vals) - 1):
        mid = (sorted_vals[i] + sorted_vals[i + 1]) / 2
        threshold.append(mid)

    return  threshold

def split_dataset(X,y,idx,threshold):
    x_left = []
    y_left = []

    x_right = []
    y_right = []

    for i in range(len(X)):

        x_i = X[i]
        y_i = y[i]

        if x_i[idx] < threshold:
            x_left.append(x_i)
            y_left.append(y_i)
        else:
            x_right.append(x_i)
            y_right.append(y_i)

    return x_left, y_left, x_right, y_right

def information_gain(parent_y, left_y, right_y):

    if len(left_y) == 0 or len(right_y) == 0:
        return 0

    parent_gini = gini(parent_y)
    left_gini = gini(left_y)
    right_gini = gini(right_y)

    left_weight = len(left_y) / len(parent_y)
    right_weight = len(right_y) / len(parent_y)

    children_gini = (
        left_weight * left_gini
        +
        right_weight * right_gini
    )

    return parent_gini - children_gini

def find_best_split(X,y):
    best_gain = -1
    best_feature = None
    best_threshold = None

    num_features = len(X[0])

    for feature_idx in range(num_features):
        feature_values = []
        for row in X:
            feature_values.append(row[feature_idx])

        thresholds = get_threshold(feature_values)
        for threshold in thresholds:
            x_left, y_left, x_right, y_right = split_dataset(X,y,feature_idx,threshold)
            gain = information_gain(y, y_left, y_right)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold

    return best_feature, best_threshold,best_gain

def build_tree(X, y):

    if len(set(y)) == 1:
        return LeafNode(y[0])

    best_feature,best_threshold,best_gain = find_best_split(X,y)
    if best_gain <= 0:
        return LeafNode(
            majority_class(y)
        )

    x_left,y_left,x_right,y_right = split_dataset(X,y,best_feature,best_threshold)

    left_tree = build_tree(x_left,y_left)
    right_tree = build_tree(x_right,y_right)

    return DecisionNode(best_feature,best_threshold,left_tree,right_tree)

def majority_class(y):
    votes = {}
    for label in y:
        votes[label] = votes.get(label, 0) + 1

    return max(votes, key=votes.get)

def predict(node, sample):

    if isinstance(node, LeafNode):
        return node.value

    if sample[node.feature_idx] < node.threshold:
        return predict(node.left, sample)

    return predict(node.right, sample)

def print_tree(node, depth=0):

    prefix = "  " * depth

    if isinstance(node, LeafNode):
        print(f"{prefix}Leaf: {node.value}")
        return

    print(
        f"{prefix}Feature[{node.feature_idx}] < {node.threshold}"
    )

    print_tree(node.left, depth + 1)
    print_tree(node.right, depth + 1)

X = [
    [1],
    [2],
    [3],
    [10],
    [11],
    [12]
]

y = [
    "A",
    "A",
    "A",
    "B",
    "B",
    "B"
]
tree = build_tree(X,y)
print_tree(tree)