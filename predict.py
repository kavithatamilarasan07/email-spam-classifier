import re
import string
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords (only first time)
nltk.download("stopwords")

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

# -----------------------------
# Text Cleaning
# -----------------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove stopwords & stem
    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -----------------------------
# Prediction Function
# -----------------------------
def predict_email(email):

    email = clean_text(email)

    email_vector = vectorizer.transform([email])

    prediction = model.predict(email_vector)

    if prediction[0] == 1:
        return "Spam ❌"
    else:
        return "Ham ✅"

# -----------------------------
# Main Program
# -----------------------------
print("=" * 50)
print("Email Spam Classifier")
print("=" * 50)

while True:

    print("\nEnter Email Text")
    print("-" * 50)

    email = input()

    result = predict_email(email)

    print("\nPrediction :", result)

    choice = input("\nDo you want to test another email? (y/n): ")

    if choice.lower() != "y":
        break

print("\nThank You!")