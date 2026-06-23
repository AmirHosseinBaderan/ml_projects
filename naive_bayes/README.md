# Naive Bayes Text Classification

## What it does

This project implements a text classification system using the Naive Bayes algorithm. It includes two implementations:

1. **main.py** - A text classification model trained on the 20 Newsgroups dataset, which classifies news articles into different topic categories using TF-IDF vectorization and Multinomial Naive Bayes.

2. **test_project.py** - A simple demonstration of spam/ham email classification using a small custom dataset with CountVectorizer.

## Implementation

### main.py
- Uses `sklearn.datasets.fetch_20newsgroups` to load the 20 Newsgroups dataset
- Applies `TfidfVectorizer` to convert text documents into numerical feature vectors
- Implements `MultinomialNB` (Multinomial Naive Bayes) classifier from scikit-learn
- Splits data into training (80%) and testing (20%) sets
- Evaluates model performance using accuracy score

### test_project.py
- Demonstrates a simple spam detection example with hand-crafted training data
- Uses `CountVectorizer` for text-to-numerical conversion
- Trains a MultinomialNB model on spam/ham email examples
- Makes predictions on test messages: "Win Iphone now" and "see you tomorrow"

## How to run

```bash
# Run the main 20 Newsgroups classification
python main.py

# Run the spam detection demo
python test_project.py
```

## Requirements

- scikit-learn
- numpy