# DBSCAN Clustering (From Scratch)

## What it does

This project implements DBSCAN (Density-Based Spatial Clustering of Applications with Noise) clustering from scratch without using scikit-learn. It groups data points into clusters based on density, identifying core points, border points, and noise points.

## Implementation

### DBSCAN.py
- Implements `euclidean_distance(p1, p2)` function to calculate distance between two points
- Implements `DBSCAN` class with:
  - `__init__(eps, min_samples)` - Initialize with epsilon (neighborhood radius) and minimum samples
  - `region_query(X, point_index)` - Find all points within eps distance of a point
  - `expand_cluster(X, point_index, neighbors, cluster_id)` - Expand cluster from a core point
  - `fit(X)` - Run the DBSCAN algorithm
  - `fit_predict(X)` - Fit and return cluster labels

### main.py
- Creates sample 2D data points with 3 natural clusters and one outlier
- Trains the custom DBSCAN model with eps=2, min_samples=2
- Prints cluster labels (-1 indicates noise/outlier)

## How to run

```bash
python main.py
```

## Output

The script prints cluster labels for each data point:
- Points in the same cluster get the same label
- Outlier points (like [25, 25]) get label -1

## Requirements

- No external ML libraries required
- Uses only Python standard library (math)