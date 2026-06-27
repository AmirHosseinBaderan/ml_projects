# ml_projects

A collection of machine learning projects implemented in Python using scikit-learn and TensorFlow.

## Projects

### Classification Algorithms
- **[naive_bayes](naive_bayes/README.md)** - Text classification using Naive Bayes (20 Newsgroups dataset and spam detection)
- **[svm](svm/README.md)** - Support Vector Machine classification for breast cancer diagnosis
- **[knn-from-scratch](knn-from-scratch/README.md)** - K-Nearest Neighbors implementation from scratch
- **[iris-from-scratch](iris-from-scratch/README.md)** - Iris flower classification with custom KNN
- **[random-forest-scratch](random-forest-scratch/README.md)** - Random Forest implementation from scratch
- **[random-forest-sklearn](random-forest-sklearn/README.md)** - Random Forest using scikit-learn

### Deep Learning & Neural Networks
- **[tensorflow-mnist](tensorflow-mnist/README.md)** - MNIST digit recognition with TensorFlow
- **[dog_cat](dog_cat/README.md)** - Dog vs cat image classification

### Natural Language Processing
- **[mini_chatbot](mini_chatbot/README.md)** - Simple rule-based chatbot using TF-IDF and cosine similarity
- **[comment_sentiment](comment_sentiment/README.md)** - Sentiment analysis for product comments

### Recommendation Systems
- **[movie](movie/README.md)** - Movie recommendation engine
- **[movie_knn](movie_knn/README.md)** - Movie recommendations using KNN

### Reinforcement Learning
- **[reinforcement-learning](reinforcement-learning/README.md)** - RL implementations
- **[tehran-metro-rl](tehran-metro-rl/README.md)** - Tehran Metro route optimization with RL

### Decision Trees
- **[decision-tree-scratch](decision-tree-scratch/README.md)** - Decision tree implementation from scratch

### Clustering
- **[k_means](k_means/README.md)** - K-Means clustering using scikit-learn
- **[k_means_from_scratch](k_means_from_scratch/README.md)** - K-Means clustering implemented from scratch
- **[hierarchical_clustering](hierarchical_clustering/README.md)** - Hierarchical clustering implemented from scratch
- **[DBSCAN_sklearn](DBSCAN_sklearn/README.md)** - DBSCAN clustering using scikit-learn
- **[DBSCAN_scratch](DBSCAN_scratch/README.md)** - DBSCAN clustering implemented from scratch

### Dimensionality Reduction
- **[PCA](PCA/README.md)** - Principal Component Analysis (custom and sklearn implementations)

## Requirements

All projects use Python with the following main dependencies:
- scikit-learn
- numpy
- tensorflow (for tensorflow-mnist)
- pillow (for dog_cat image processing)
- matplotlib (for visualization projects)

Install dependencies:
```bash
pip install -r requirements.txt