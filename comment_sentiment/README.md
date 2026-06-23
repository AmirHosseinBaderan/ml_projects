# Comment Sentiment Analysis

## What it does

This project implements a sentiment analysis system that classifies user comments into three categories: positive, negative, or neutral. It uses a Naive Bayes classifier trained on a custom dataset of product review comments.

## Implementation

### main.py
- Uses `TfidfVectorizer` from scikit-learn to convert text comments into numerical feature vectors
- Implements `MultinomialNB` (Multinomial Naive Bayes) classifier for sentiment classification
- Runs an interactive loop that accepts user comments and predicts their sentiment
- Type "exit" to stop the program

### data.py
- Contains a predefined list of 40+ question-answer pairs for training
- Training data includes:
  - Positive comments (e.g., "i love this product", "this is amazing", "best purchase ever")
  - Negative comments (e.g., "this is bad", "i hate it", "worst experience ever")
  - Neutral comments (e.g., "it's ok", "average quality", "does the job")

## How to run

```bash
python main.py
```

## Example interaction

```
write comment : i love this product
predicted sentiment:  positive
write comment : this is terrible
predicted sentiment:  negative
write comment : it's okay
predicted sentiment:  neutral
write comment : exit
goodbye
```

## Requirements

- scikit-learn
- numpy