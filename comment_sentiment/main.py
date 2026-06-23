from sklearn.feature_extraction.text import  TfidfVectorizer
from data import data
from sklearn.naive_bayes import MultinomialNB

texts = [x[0] for x in data]
labels = [x[1] for x in data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

while True:
    user_input = input("write comment : ")
    if user_input == 'exit':
        print('goodbye')
        break

    X_test = vectorizer.transform([user_input])
    preds = model.predict(X_test)

    print("predicted sentiment: ", preds[0])