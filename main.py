import os
import json
import sys

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import main functions from pipeline stages
from src.data_generation import main as generate_data_main
from src.nlp_train import main as train_model_main
from src.nlp_predict import predict as predict_text
from src.clean import clean_text # Import clean_text for consistency, though predict_text uses it internally

def main():
    print("Starting the full pipeline orchestration...")

    # Define paths
    SYNTHETIC_DATA_PATH = "data/synthetic_posts.json"
    MODEL_PATH = "models/logistic_regression_model.pkl"

    # Step 1: Generate Data if not exists
    if not os.path.exists(SYNTHETIC_DATA_PATH):
        print(f"--- Generating Sentiment Data: {SYNTHETIC_DATA_PATH} not found ---")
        generate_data_main()
    else:
        print(f"--- {SYNTHETIC_DATA_PATH} already exists. Skipping data generation. ---")

    # Step 2: Clean Data (from synthetic_posts.json)
    CLEANED_CSV_PATH = "data/cleaned_sentiment_data.csv"
    CLEANED_JSON_PATH = "data/cleaned_sentiment_data.json"
    if not os.path.exists(CLEANED_CSV_PATH) or not os.path.exists(CLEANED_JSON_PATH):
        print(f"--- Cleaning Data: {CLEANED_CSV_PATH} or {CLEANED_JSON_PATH} not found ---")
        # Use subprocess to run clean.py as it's designed to be run as a script
        import subprocess
        subprocess.run(["python3", "-m", "src.clean", "--input_file", SYNTHETIC_DATA_PATH, "--output_dir", "data"], check=True)
    else:
        print(f"--- Cleaned data already exists. Skipping cleaning. ---")

    # Step 2: Train Model if not exists
    if not os.path.exists(MODEL_PATH):
        print(f"--- Training NLP Model: {MODEL_PATH} not found ---")
        train_model_main()
    else:
        print(f"--- {MODEL_PATH} already exists. Skipping model training. ---")

    # Step 3: Test Prediction
    print("--- Testing Prediction ---")
    test_texts = [
        "This is a neutral statement about the weather.",
        "The government's policies are a complete disaster for the country.",
        "I love the diversity of cultures in India.",
        "The situation in Kashmir is a human rights crisis. #KashmirBleeds",
        "This is a test of a neutral sentence with no keywords."
    ]

    for text in test_texts:
        print(f"Predicting for: \"{text}\"")
        prediction = predict_text(text)
        print(f"Prediction: {prediction}\n")

    print("Full pipeline orchestration completed successfully!")

if __name__ == "__main__":
    main()
