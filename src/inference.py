import os
import joblib

class AntiIndiaDetector:
    def __init__(self, model_path='models/anti_india_detector/'):
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        """
        Load the trained model from specified path
        """
        try:
            model = joblib.load(os.path.join(model_path, 'model.pkl'))
            print(f"Model loaded successfully from {model_path}")
            return model
        except FileNotFoundError:
            print(f"Model file not found at {model_path}")
            raise

    def predict(self, tweet: str) -> str:
        """
        Predict whether a tweet contains anti-India sentiment
        Returns: 'flagged' or 'neutral'
        """
        prediction = self.model.predict([tweet])[0]
        return 'flagged' if prediction == 'flagged' else 'neutral'

if __name__ == '__main__':
    # Example usage
    detector = AntiIndiaDetector()
    test_tweets = [
        "This is a neutral tweet",
        "This tweet contains anti-India content"
    ]
    
    for tweet in test_tweets:
        print(f"Tweet: '{tweet}' -> Prediction: {detector.predict(tweet)}")