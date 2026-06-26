# K-Means Clustering (From Scratch)

## What it does

This project implements the K-Means clustering algorithm from scratch without using scikit-learn. It groups customer data into clusters based on order frequency and spending patterns, demonstrating the core algorithm mechanics.

## Implementation

### k_mean.py
- Implements `euclidean_distance(p1, p2)` function to calculate distance between two points
- Implements `KMeans` class with:
  - `__init__(k, max_iters)` - Initialize with k clusters and max iterations
  - `init_centroids(X)` - Randomly select k data points as initial centroids
  - `assign_clusters(X)` - Assign each point to the nearest centroid
  - `update_centroids(clusters)` - Recalculate centroids as mean of assigned points
  - `fit(X)` - Run the K-Means algorithm until convergence or max iterations
  - `predict(point)` - Predict cluster for a new data point

### main.py
- Creates sample customer data with orders and spend values
- Trains the custom KMeans model
- Prints cluster centroids and cluster assignments
- Demonstrates prediction on new data points

## How to run

```bash
python main.py
```

## Output

The script prints:
- Cluster centroids (center points of each cluster)
- Cluster assignments for training data
- Predictions for test points: [95, 9000] and [2, 120]

## Requirements

- No external ML libraries required
- Uses only Python standard library (math, random)