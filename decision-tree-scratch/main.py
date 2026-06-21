
def gini(y):
    counts = {}

    for label in y:
        counts[label] = counts.get(label, 0) + 1

    impurity = 0
    total = len(y)

    for label in counts:
        p = counts[label] / total
        impurity -= p ** 2

    return impurity

def get_threshold(feature_column):
    sorted_vals = sorted(feature_column)

    threshold = []

    for i in range(len(sorted_vals) - 1):
        mid = (sorted_vals[i] + sorted_vals[i + 1]) / 2