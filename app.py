from flask import Flask, render_template, request
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
nltk.download("stopwords")

app = Flask(__name__)

# Load Model & Vectorizer
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)

    # Remove Numbers
    text = re.sub(r"\d+", "", text)

    # Remove Punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove Stopwords & Stemming
    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Prediction
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    email = request.form["email"]

    cleaned_email = clean_text(email)

    vector = vectorizer.transform([cleaned_email])

    prediction = model.predict(vector)

    if prediction[0] == 1:
        result = "Spam ❌"
    else:
        result = "Ham ✅"

    return render_template(
        "index.html",
        prediction=result,
        email=email
    )

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)