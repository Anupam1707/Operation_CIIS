from flask import Flask, request, jsonify
import sqlite3
from datetime import datetime
from pathlib import Path
from .inference import classify_tweet

app = Flask(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "tweets.db"

def store_classified_tweet(tweet_text: str, label: str):
    """
    Stores the classified tweet and its label in the database.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            # Assuming a table 'classified_tweets' exists.
            # If not, this will fail. A setup script should create it.
            # For now, let's ensure the table exists.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS classified_tweets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tweet_text TEXT NOT NULL,
                    label TEXT NOT NULL,
                    classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute(
                "INSERT INTO classified_tweets (tweet_text, label, classified_at) VALUES (?, ?, ?)",
                (tweet_text, label, datetime.now())
            )
            conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        # Decide how to handle DB errors; for now, we just print.
        # In a production system, you might want to log this to a file or monitoring service.


@app.route('/classify', methods=['POST'])
def classify():
    """
    API endpoint to classify a tweet.
    Accepts a JSON payload with a "tweet" key.
    e.g., {"tweet": "some text to classify"}
    """
    if not request.json or 'tweet' not in request.json:
        return jsonify({"error": "Invalid input. JSON with 'tweet' key is required."}), 400

    tweet_text = request.json['tweet']
    if not isinstance(tweet_text, str) or not tweet_text.strip():
        return jsonify({"error": "Invalid input. 'tweet' must be a non-empty string."}), 400

    try:
        # Get the classification label
        label = classify_tweet(tweet_text)

        # Store the tweet and its label in the database
        store_classified_tweet(tweet_text, label)

        # Return the result
        return jsonify({"tweet": tweet_text, "label": label})

    except Exception as e:
        # Generic error handler for unexpected issues during classification
        print(f"An error occurred during classification: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == '__main__':
    # To run this app:
    # 1. Make sure you are in the root directory 'Operation_CIIS'
    # 2. Run the app using: flask --app src.app run
    # Example POST request using curl:
    # curl -X POST -H "Content-Type: application/json" -d "{\"tweet\":\"This is a test tweet.\"}" http://127.0.0.1:5000/classify
    app.run(debug=True)
