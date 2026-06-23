from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = [
    ("win money now", "spam"),
    ("free lottery prize", "spam"),
    ("click this link", "spam"),
    ("urgent win iphone", "spam"),

    ("let's meet tomorrow", "ham"),
    ("how are you", "ham"),
    ("are you coming home", "ham"),
    ("see you at school", "ham"),
]

texts = [t[0] for t in data]
labels = [t[1] for t in data]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# naive bayes
model = MultinomialNB()
model.fit(X, labels)

test = vectorizer.transform(["Win Iphone now","see you tomorrow"])
print(model.predict(test))