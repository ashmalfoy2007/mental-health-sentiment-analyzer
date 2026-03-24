# -------------------------------
# Mental Health Sentiment Analyzer
# -------------------------------

# step 1: import libraries
import pandas as pd
import string
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# download stopwords (only first time)
nltk.download('stopwords')

# step 2: load dataset
# make sure your dataset has columns: 'text' and 'emotion'
# step 2: load dataset
data = pd.read_csv("dataset.csv", sep=';')

# remove missing values (VERY IMPORTANT)
data = data.dropna(subset=['text', 'emotion'])

# step 3: text preprocessing
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

data['clean_text'] = data['text'].apply(preprocess)

# step 4: split data
X = data['clean_text']
y = data['emotion']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# step 5: convert text to numbers (TF-IDF)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# step 6: train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# step 7: evaluate model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# -------------------------------
# prediction + suggestion system
# -------------------------------

def suggest(emotion):
    suggestions = {
        "sad": "Talk to a friend or listen to calming music.",
        "happy": "Keep doing what makes you happy!",
        "anger": "Take deep breaths and give yourself time to relax.",
        "fear": "Try to focus on positive thoughts and stay calm.",
        "love": "Share your feelings with loved ones.",
        "surprise": "Take a moment to process and enjoy the moment."
    }
    return suggestions.get(emotion, "Take care of yourself and stay positive.")

def predict_emotion(text):
    clean = preprocess(text)
    vec = vectorizer.transform([clean])

    prediction = model.predict(vec)[0]
    probabilities = model.predict_proba(vec)[0]

    confidence = max(probabilities) * 100

    return prediction, confidence

# -------------------------------
# user interaction loop
# -------------------------------

print("\n--- Mental Health Sentiment Analyzer ---")

while True:
    user_input = input("\nEnter your feelings (or type 'exit'): ")

    if user_input.lower() == "exit":
        print("Take care! 💙")
        break

    emotion, confidence = predict_emotion(user_input)
    advice = suggest(emotion)

    print("\nDetected Emotion:", emotion)
    print("Confidence:", round(confidence, 2), "%")
    print("Suggestion:", advice)