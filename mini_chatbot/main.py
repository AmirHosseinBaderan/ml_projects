from sklearn.feature_extraction.text import TfidfVectorizer
from data import data
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

questions = [x[0] for x in data]
answers = [x[1] for x in data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)


# chat bot function
def get_response(user_input):
    user_vec = vectorizer.transform([user_input])

    similarities = cosine_similarity(user_vec, X)

    best_idx = np.argmax(similarities)

    if similarities[0][best_idx] < 0.2:
        return "I dont understand"

    return answers[best_idx]

# chat bot loop
print("Bot is running... (type exit to stop)")

while True:
    msg = input("You: ")

    if msg == "exit":
        break

    response = get_response(msg)
    print("Bot:", response)