import os
import re
import string
import joblib
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# Download NLTK data
# -----------------------------
nltk.download("stopwords")

# -----------------------------
# Load Dataset
# -----------------------------
DATASET_PATH = "dataset/spam.csv"

if not os.path.exists(DATASET_PATH):
    print("Dataset not found!")
    print("Place spam.csv inside dataset folder.")
    exit()

df = pd.read_csv(DATASET_PATH, encoding="latin-1")

# Keep only required columns
df = df.iloc[:, :2]
df.columns = ["label", "message"]

# Convert labels
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# -----------------------------
# Text Cleaning
# -----------------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"www\\S+", "", text)
    text = re.sub(r"\\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["message"] = df["message"].apply(clean_text)

# -----------------------------
# TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["message"])

y = df["label"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = MultinomialNB()

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("=" * 50)
print("Email Spam Classifier")
print("=" * 50)

print(f"Accuracy : {accuracy * 100:.2f}%")

print("\nClassification Report\n")

print(classification_report(
    y_test,
    y_pred
))

print("\nConfusion Matrix\n")

print(confusion_matrix(
    y_test,
    y_pred
))

# -----------------------------
# Save Model
# -----------------------------
os.makedirs("model", exist_ok=True)

joblib.dump(
    model,
    "model/spam_model.pkl"
)

joblib.dump(
    vectorizer,
    "model/vectorizer.pkl"
)

print("\nModel Saved Successfully!")
print("spam_model.pkl")
print("vectorizer.pkl")