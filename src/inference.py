import joblib

class AntiIndiaDetector:
    def __init__(self, model_path='models/sentiment_model.pkl', vectorizer_path='models/tfidf_vectorizer.pkl'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, tweet: str) -> str:
        """
        Predict whether a tweet contains anti-India sentiment
        Returns: 'anti-indian' or 'neutral'
        """
        tweet_tfidf = self.vectorizer.transform([tweet])
        prediction = self.model.predict(tweet_tfidf)[0]
        return prediction

if __name__ == '__main__':
    # Example usage
    detector = AntiIndiaDetector()
    test_tweets = [
        "This is a neutral tweet",
        "This tweet contains anti-India content and #ModiFailedIndia"
    ]
    
    for tweet in test_tweets:
        print(f"Tweet: '{tweet}' -> Prediction: {detector.predict(tweet)}")
