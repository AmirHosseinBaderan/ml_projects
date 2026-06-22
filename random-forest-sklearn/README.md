# Random Forest with Scikit-Learn

A simple implementation of Random Forest classifier using scikit-learn to classify iris flower species.

## What It Does

This project uses the Random Forest algorithm from scikit-learn to predict iris flower species (Setosa, Versicolor, Virginica) based on their measurements (sepal length, sepal width, petal length, petal width). It demonstrates the use of an ensemble method for classification.

## Implementation

### Data Processing

- Loads the Iris dataset from `data/iris.csv`
- Splits data into features (X) and labels (y)
- Uses 80/20 train/test split with random state 42 for reproducibility

### Model Configuration

- **RandomForestClassifier** with:
  - `n_estimators=100`: 100 decision trees in the ensemble
  - `random_state=42`: For reproducible results

### Training and Evaluation

1. Load and prepare the dataset
2. Split into training and test sets
3. Create the Random Forest model
4. Train the model on the training data
5. Make predictions on the test set
6. Calculate and print accuracy using `accuracy_score`
7. Print the predictions

## Usage

```bash
cd random-forest-sklearn
python main.py
```

## Dependencies

- `pandas`: For data loading and manipulation
- `scikit-learn`: For RandomForestClassifier, train_test_split, and accuracy_score

## Output

The program outputs:
- The classification accuracy on the test set
- The predicted labels for all test samples

## How Random Forest Works

Random Forest is an ensemble learning method that:
1. Creates multiple decision trees (100 in this case)
2. Each tree is trained on a random subset of the data (bootstrapping)
3. Each split considers a random subset of features
4. Final prediction is made by majority voting among all trees

This approach reduces overfitting and typically provides better accuracy than a single decision tree.