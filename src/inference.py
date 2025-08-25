import joblib
import os
from pathlib import Path

# Build the path to the model file
# Assuming the script is run from the root directory or the path is relative to this file's location
try:
    # This works when scripts are run from the project root
    MODEL_PATH = Path("models/anti_india_detector/model.joblib")
    if not MODEL_PATH.exists():
        # Fallback for when the script is run from within the src directory
        MODEL_PATH = Path(__file__).parent.parent / "models/anti_india_detector/model.joblib"
except NameError:
    # For environments where __file__ is not defined (e.g., some notebooks)
    MODEL_PATH = Path("models/anti_india_detector/model.joblib")


# Load the trained model pipeline
if not os.path.exists(MODEL_PATH):
    # A mock/dummy predictor if the model doesn't exist.
    # This allows the API to run without a trained model for development purposes.
    print(f"Warning: Model file not found at {MODEL_PATH}. Using a dummy classifier.")
    class DummyModel:
        def predict(self, texts):
            # Returns 'neutral' for any input
            return ['neutral'] * len(texts)
    model = DummyModel()
else:
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")

def classify_tweet(tweet_text: str) -> str:
    """
    Classifies a single tweet text.

    Args:
        tweet_text: The text of the tweet to classify.

    Returns:
        The predicted label ('flagged' or 'neutral').
    """
    # The model expects an iterable, so we pass the text in a list
    prediction = model.predict([tweet_text])
    # The prediction is an array, so we get the first element
    label = prediction[0]
    return label

if __name__ == '__main__':
    # Example usage for testing the inference script directly
    sample_tweet_1 = "This is a neutral tweet about technology."
    sample_tweet_2 = "This is a potentially problematic tweet." # Replace with a real example

    label_1 = classify_tweet(sample_tweet_1)
    label_2 = classify_tweet(sample_tweet_2)

    print(f"Tweet: '{sample_tweet_1}'\nLabel: {label_1}")
    print("-" * 20)
    print(f"Tweet: '{sample_tweet_2}'\nLabel: {label_2}")