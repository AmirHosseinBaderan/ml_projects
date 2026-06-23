from sklearn.naive_bayes import MultinomialNB

from data import data
from sklearn.feature_extraction.text import  CountVectorizer

texts = [x[0] for x in data]
labels = [x[1] for x in data]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

print("Chatbot is running... (type 'exit' to stop)")

while True:
    user_input = input("You: ")

    if user_input == "exit":
        break

    x = vectorizer.transform([user_input])
    intent = model.predict(x)[0]

    if intent == "greeting":
        print("Bot: Hello ")

    elif intent == "goodbye":
        print("Bot: Bye ")

    elif intent == "thanks":
        print("Bot: You're welcome ")

    elif intent == "spam":
        print("Bot:  suspicious message detected")

    elif intent == "question":
        print("Bot: I'm a simple ML bot ")

    else:
        print("Bot: I don't understand ")