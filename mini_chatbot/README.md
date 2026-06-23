# Mini Chatbot

## What it does

This project implements a simple rule-based chatbot that responds to user input using TF-IDF vectorization and cosine similarity. The bot can handle greetings, farewells, questions about AI, jokes, fun facts, and can detect spam messages.

## Implementation

### main.py
- Uses `TfidfVectorizer` from scikit-learn to convert user questions into numerical vectors
- Implements a `get_response()` function that:
  - Transforms user input into a vector
  - Computes cosine similarity between the input and all known questions
  - Returns the answer with the highest similarity score
  - Returns "I dont understand" if similarity is below 0.2 threshold
- Runs an interactive chat loop that accepts user input until "exit" is typed

### data.py
- Contains a predefined list of question-answer pairs
- Covers various intents:
  - Greetings (hi, hello, hey, good morning/afternoon/evening)
  - Farewells (bye, goodbye, see you, good night)
  - Thanks responses
  - AI-related questions
  - Jokes and fun facts
  - Help requests
  - Spam detection responses
  - Love expressions

## How to run

```bash
python main.py
```

## Example interaction

```
Bot is running... (type exit to stop)
You: hello
Bot: Hi there 👋
You: tell me a joke
Bot: Why did the scarecrow win an award? Because he was outstanding in his field! 🌾
You: exit
```

## Requirements

- scikit-learn
- numpy