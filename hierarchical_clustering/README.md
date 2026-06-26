# Hierarchical Clustering (From Scratch)

## What it does

This project implements hierarchical clustering from scratch without using scikit-learn. It groups data points into clusters using a bottom-up agglomerative approach, merging the closest clusters iteratively until the desired number of clusters is reached.

## Implementation

### hirerachical.py
- Implements `euclidean_distance(p1, p2)` function to calculate distance between two points
- Implements `HierarchicalClustering` class with:
  - `cluster_distance(cluster1, cluster2)` - Calculates minimum distance between any two points in different clusters
  - `find_closest_cluster()` - Finds the pair of clusters with minimum distance
  - `merge_clusters(i, j)` - Merges two clusters at indices i and j
  - `fit(X, n_clusters)` - Runs the agglomerative clustering algorithm
  - `get_labels()` - Returns cluster labels for each data point

### main.py
- Creates sample 1D data points: [1, 2, 10, 11]
- Trains the HierarchicalClustering model with 2 clusters
- Prints the resulting clusters and their labels

## How to run

```bash
python main.py
```

## Output

The script prints:
- The final clusters (groups of data points)
- Cluster labels for each data point

## Requirements

- No external ML libraries required
- Uses only Python standard library (math)