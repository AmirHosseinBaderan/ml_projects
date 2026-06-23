from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# data
data = fetch_20newsgroups(subset='all')
X, y = data.data, data.target

# vectorization (text -> number)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# model
model = MultinomialNB()
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred)*100}%")
