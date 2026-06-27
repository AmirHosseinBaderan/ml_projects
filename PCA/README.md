# Principal Component Analysis (PCA)

## What it does

This project implements PCA (Principal Component Analysis) for dimensionality reduction. It includes both a custom implementation from scratch and a comparison with scikit-learn's implementation.

## Implementation

### pca.py (Custom Implementation)
- Implements `PCA` class with:
  - `__init__(n_components)` - Initialize with number of components to keep
  - `fit(X)` - Compute principal components using:
    - Mean centering of data
    - Covariance matrix calculation
    - Eigenvalue decomposition
    - Sorting and selecting top components
  - `transform(X)` - Project data onto principal components
  - `fit_transform(X)` - Fit and transform in one step
  - `inverse_transform(X)` - Reconstruct original data from reduced dimensions
  - `explained_variance_ratio_` - Ratio of variance explained by each component

### main.py (Custom PCA)
- Creates sample height/weight data
- Standardizes the data using StandardScaler
- Applies custom PCA to reduce to 1 component
- Prints transformed data, explained variance ratio, and reconstructed data

### sk_learn/main.py (scikit-learn comparison)
- Same sample data as main.py
- Uses `sklearn.decomposition.PCA` for comparison
- Reduces to 2 components and prints results

## How to run

```bash
# Run custom PCA implementation
python main.py

# Run scikit-learn comparison
python sk_learn/main.py
```

## Output

- Transformed data in reduced dimensional space
- Explained variance ratio showing information retention
- Reconstructed data (inverse transform)

## Requirements

- numpy
- scikit-learn (for StandardScaler and sk_learn comparison)