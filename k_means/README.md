# K-Means Clustering (scikit-learn)

## What it does

This project implements customer segmentation using K-Means clustering algorithm from scikit-learn. It groups customers into clusters based on their order frequency and spending patterns.

## Implementation

- Uses `sklearn.cluster.KMeans` for clustering
- Uses `pandas` to create a DataFrame with customer data
- Features used for clustering:
  - `orders` - Number of orders placed
  - `spend` - Amount spent
- Configures the model with:
  - `n_clusters=3` - Divides customers into 3 groups
  - `random_state=42` - For reproducible results
- Outputs cluster assignments and cluster centers

## How to run

```bash
python main.py
```

## Output

The script prints:
- Cluster labels for each customer
- Coordinates of the cluster centers

## Requirements

- scikit-learn
- pandas
- numpy