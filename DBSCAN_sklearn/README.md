# DBSCAN Clustering (scikit-learn)

## What it does

This project implements DBSCAN (Density-Based Spatial Clustering of Applications with Noise) clustering using scikit-learn. It groups data points into clusters based on density, identifying core points, border points, and noise points.

## Implementation

- Uses `sklearn.cluster.DBSCAN` for clustering
- Uses `matplotlib` to visualize the data and clustering results
- Sample 2D data points with 3 natural clusters:
  - Cluster 1: [1, 2], [2, 2], [2, 3]
  - Cluster 2: [8, 8], [8, 9]
  - Outlier: [25, 25]
- Configures the model with:
  - `eps=2` - Maximum distance between two points to be considered neighbors
  - `min_samples=2` - Minimum number of points to form a dense region

## How to run

```bash
python main.py
```

## Output

The script displays:
- Initial scatter plot of data points
- Final scatter plot with points colored by cluster labels
- Cluster labels printed to console (-1 indicates noise/outlier)

## Requirements

- scikit-learn
- matplotlib
- numpy