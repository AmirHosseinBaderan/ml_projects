# Support Vector Machine (SVM) Classification

## What it does

This project implements a binary classification model using Support Vector Machine (SVM) to predict whether breast cancer tumors are malignant or benign. It uses the Wisconsin Breast Cancer dataset from scikit-learn and demonstrates the application of SVM with an RBF kernel for medical diagnosis.

## Implementation

- Uses `sklearn.datasets.load_breast_cancer` to load the Wisconsin Breast Cancer dataset
- Implements `SVC` (Support Vector Classification) with RBF kernel from scikit-learn
- Configures the model with:
  - `kernel='rbf'` - Radial Basis Function kernel for non-linear classification
  - `C=1.0` - Regularization parameter for balancing margin and misclassification
- Splits data into training (80%) and testing (20%) sets with random state 42
- Trains the SVM model on the training data
- Makes predictions on the test set
- Evaluates model performance using accuracy score

## How to run

```bash
python main.py
```

## Output

The script will print the accuracy of the SVM model on the test set, showing the percentage of correctly classified tumor samples.

## Requirements

- scikit-learn
- numpy