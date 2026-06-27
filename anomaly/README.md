# Anomaly Detection with Isolation Forest

This project implements **anomaly detection** using the **Isolation Forest** algorithm. It provides two implementations: a from-scratch implementation for educational purposes and a comparison using scikit-learn's built-in implementation.

## What It Does

Anomaly detection identifies rare items, events, or observations that differ significantly from the majority of the data. This project uses **Isolation Forest**, an unsupervised learning algorithm that isolates anomalies by randomly selecting features and split values.

The algorithm works on the principle that anomalies are:
- **Few** in number
- **Different** in feature values

These properties make anomalies easier to isolate with fewer random splits in a decision tree.

## Project Structure

```
anomaly/
├── main.py              # From-scratch Isolation Forest demo (1D and 2D examples)
├── isolation_forest.py  # Core Isolation Forest implementation
├── node.py              # Binary tree node class
├── sklearn/
│   └── main.py          # Scikit-learn Isolation Forest comparison
└── README.md            # This file
```

## Implementation Details

### From-Scratch Implementation (`isolation_forest.py`)

The [`IsolationForest`](anomaly/isolation_forest.py:7) class implements the algorithm from scratch:

- **`fit(X)`** — Builds an ensemble of isolation trees by randomly sampling subsets of data and recursively splitting on random features and split values.
- **`predict(X)`** — Computes anomaly scores for each sample by averaging path lengths across all trees. Samples with scores above `0.7` are classified as anomalies.
- **`_path_length(x, node, depth)`** — Recursively traverses a tree to compute the path length of a sample.
- **`_build_tree(X, depth)`** — Recursively constructs an isolation tree by selecting random features and random split values.
- **`_c(n)`** — Computes the expected path length correction factor for a sample of size `n`, based on the harmonic number approximation.

### Node Class (`node.py`)

The [`Node`](anomaly/node.py:2) class represents a binary tree node with:
- `feature` — The feature index used for splitting
- `split_value` — The threshold value for the split
- `left` / `right` — Child nodes
- `size` — Number of samples in the node (used for leaf nodes)

### Scikit-learn Comparison (`sklearn/main.py`)

The [`sklearn/main.py`](anomaly/sklearn/main.py:1) script demonstrates the same anomaly detection task using scikit-learn's [`IsolationForest`](anomaly/sklearn/main.py:2) for comparison, showing predictions, anomaly scores, and the model's offset parameter.

## How to Run

### From-Scratch Demo

```bash
cd anomaly
python main.py
```

This runs two examples:
1. **1D Example** — 8 samples where 2 are anomalies (values 90, 95)
2. **2D Example** — 100 normal samples (Gaussian distribution) + 3 anomalies

### Scikit-learn Comparison

```bash
cd anomaly/sklearn
python main.py
```

## Algorithm Overview

1. **Build Phase**: For each tree in the ensemble:
   - Randomly sample a subset of the data
   - Recursively split by selecting a random feature and a random split value between min and max
   - Stop when all samples are isolated or max depth is reached

2. **Score Phase**: For each sample:
   - Compute the average path length across all trees
   - Convert to an anomaly score using: `score = 2^(-avg_path / c(n))`
   - Where `c(n)` is the expected path length correction

3. **Classification**: Samples with higher scores are more likely to be anomalies.

## Dependencies

- Python 3.x
- NumPy
- scikit-learn (for the comparison script only)
