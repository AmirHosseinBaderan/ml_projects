# KNN from Scratch

## What it does
Implements a K-Nearest Neighbors (KNN) classifier from scratch using only Python's standard library (`math`, `collections`, `operator`) and `pandas`. Trains and evaluates on the Iris dataset.

## How it works
1. Loads the Iris dataset from a CSV file
2. Splits features (`X`) and target labels (`y`)
3. Computes `euclidean_distance` between a sample point and all training points
4. Finds the `k` nearest neighbors by sorting distances
5. Predicts the class using majority voting (`Counter.most_common`)
6. Evaluates accuracy by predicting each sample and comparing to the true label

## Implementation
- `knn.py`:
  - `euclidean_distance(a, b)` — computes L2 distance between two feature vectors
  - `get_neighbors(X, y, sample, k)` — returns `k` closest `(vector, label, distance)` tuples
  - `predict(X, y, sample, k)` — majority-vote prediction from neighbors
- `main.py` — loads Iris data, runs neighbor lookup, distance computation, and leaves-one-out accuracy check

## Run
```bash
python main.py
```
