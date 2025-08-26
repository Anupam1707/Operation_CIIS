import argparse
import os
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources if not already present
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.Downloader.DownloadError:
    nltk.download('punkt')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

import argparse
import os
import joblib

from src.clean import clean_text # Import clean_text from clean.py

def preprocess_text(text):
    text = text.lower()
    # Use regex to find words, bypassing nltk.word_tokenize's punkt dependency
    tokens = re.findall(r'\b\w+\b', text)
    tokens = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

def predict(text: str):
    MODEL_PATH = "models"
    model = joblib.load(os.path.join(MODEL_PATH, "logistic_regression_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"))
    processed_text = clean_text(text)
    text_vec = vectorizer.transform([processed_text])
    
    prediction = model.predict(text_vec)[0]
    # For Logistic Regression, decision_function gives confidence scores
    # For binary classification, it's usually a single value,
    # and its magnitude indicates confidence.
    # We can convert it to probabilities using sigmoid if needed,
    # but for simplicity, we'll use the absolute value or just the raw decision.
    
    # To get probabilities:
    probabilities = model.predict_proba(text_vec)[0]
    confidence = float(probabilities[prediction])

    return {
        "text": text,
        "label": "anti-indian" if prediction == 1 else "neutral",
        "confidence": confidence
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict anti-Indian content in a given text.")
    parser.add_argument("--text", type=str, required=True, help="The text to classify.")
    args = parser.parse_args()

    prediction = predict(args.text)
    print(prediction)
