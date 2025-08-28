import pandas as pd
import json
import os
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import joblib # For saving/loading models and vectorizers

from src.clean import clean_text # Import clean_text from clean.py

# Compute metrics
def compute_metrics(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def run_training():
    # Load synthetic data
    with open("data/synthetic_posts.json", "r") as f:
        synthetic_data = json.load(f)

    processed_synthetic_data = []
    for post in synthetic_data:
        processed_synthetic_data.append({
            "text": clean_text(post["text"]), # Preprocess text using clean.py
            "label": 1 if post["label"] == "anti-indian" else 0
        })

    synthetic_df = pd.DataFrame(processed_synthetic_data)

    X = synthetic_df["text"]
    y = synthetic_df["label"]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=5000) # Limit features to avoid sparsity
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Load model
    model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence

    print("Training Logistic Regression model...")
    model.fit(X_train_vec, y_train)

    # Evaluate and log metrics
    y_pred = model.predict(X_test_vec)
    metrics = compute_metrics(y_test, y_pred)
    print("Evaluation Metrics:", metrics)

    # Save trained model and vectorizer
    model_path = "models"
    os.makedirs(model_path, exist_ok=True)
    joblib.dump(model, os.path.join(model_path, "logistic_regression_model.pkl"))
    joblib.dump(vectorizer, os.path.join(model_path, "tfidf_vectorizer.pkl"))
    print(f"Saved model and vectorizer to {model_path}")

    # Generate evaluation report
    report_path = "reports"
    os.makedirs(report_path, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "overall": {
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1")
            }
        }
    }
    with open(f"{report_path}/step2_metrics.json", "w") as f:
        json.dump(report, f, indent=4)
    print(f"Saved metrics to {report_path}/step2_metrics.json")

if __name__ == "__main__":
    run_training()

main = run_training # Expose run_training as main for import