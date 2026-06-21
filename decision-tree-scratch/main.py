
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