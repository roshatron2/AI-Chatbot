from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
import numpy as np
import random
import string
import pandas as pd


# question_answer_pairs = {
#     "When wil cat 2 be conducted": "24th September",
#     "When will Cat 1 be conducted": "21 August",
#     "When will quiz 1 be conducted": "July 1",
#     "When is DA 1 due": "July 17th",
#     "When is DA 2 due": "August 8th",
#     "When will cat 1 be conducted": "August 1st",
#     "When will fat be conducted": " December 1st",
# }

df = pd.read_csv("dataset.csv")


questions = []
for question in df["question"]:
    questions.append(question.lower())

wnlemmatizer = nltk.stem.WordNetLemmatizer()

greeting_inputs = (
    "hey",
    "good morning",
    "good evening",
    "morning",
    "evening",
    "hi",
    "whatsup",
)
greeting_responses = [
    "hey",
    "hey hows you?",
    "*nods*",
    "hello, how you doing",
    "hello",
    "Welcome, I am good and you",
]


def generate_greeting_response(greeting):
    for token in greeting.split():
        if token.lower() in greeting_inputs:
            return random.choice(greeting_responses)


def perform_lemmatization(tokens):
    return [wnlemmatizer.lemmatize(token) for token in tokens]


punctuation_removal = dict(
    (ord(punctuation), None) for punctuation in string.punctuation
)


def get_processed_text(document):
    return perform_lemmatization(
        nltk.word_tokenize(document.lower().translate(punctuation_removal))
    )


def generate_response(user_input):
    bot_response = ""
    questions.append(user_input)

    word_vectorizer = TfidfVectorizer(
        tokenizer=get_processed_text, stop_words="english"
    )
    all_word_vectors = word_vectorizer.fit_transform(questions)
    similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
    similar_sentence_number = similar_vector_values.argsort()[0][-2]

    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    if vector_matched == 0:
        bot_response = bot_response + "I am sorry, I could not understand you"
        return bot_response
    else:
        bot_response = bot_response + df["answer"][similar_sentence_number]
        return bot_response


continue_dialogue = True
print("Hello, I am your friend DateBot. You can ask me any question regarding Deadlines")
while continue_dialogue == True:
    human_text = input()
    human_text = human_text.lower()
    if human_text != "bye":
        if (
            human_text == "thanks"
            or human_text == "thank you very much"
            or human_text == "thank you"
        ):
            continue_dialogue = False
            print("DateBot: Most welcome")
        else:
            if generate_greeting_response(human_text) != None:
                print("DateBot: " + generate_greeting_response(human_text))
            else:
                print("DateBot: ", end="")
                print(generate_response(human_text))
                questions.remove(human_text)
    else:
        continue_dialogue = False
        print("DateBot: Good bye and take care of yourself...")
