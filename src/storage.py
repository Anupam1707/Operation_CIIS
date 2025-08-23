# mongo_handler.py

import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient, errors
from pymongo.operations import UpdateOne

class MongoHandler:
    """
    A handler for all MongoDB operations for the X Bot Detector project.
    Manages connection and provides CRUD operations for 'keywords' and 'tweets' collections.
    """
    def __init__(self):
        """Initializes the database connection using credentials from a .env file."""
        load_dotenv()
        mongo_uri = os.getenv("MONGO_URI")
        
        if not mongo_uri:
            raise ValueError("MONGO_URI not found in environment variables. Please check your .env file.")
            
        try:
            self.client = MongoClient(mongo_uri)
            self.client.admin.command('ping') # Confirm successful connection
            self.db = self.client.x_bot_detector_db
            self.keywords_collection = self.db.keywords
            self.tweets_collection = self.db.tweets
            # Create a unique index on the tweet ID to prevent duplicates efficiently
            self.tweets_collection.create_index("_id", unique=True)
            print("✅ MongoDB connection successful.")
        except errors.ConnectionFailure as e:
            print(f"❌ Could not connect to MongoDB: {e}")
            raise

    # --- Keyword CRUD Operations ---
    def add_keyword(self, keyword: str, category: str):
        """Adds or updates a keyword. Uses the keyword as the unique _id."""
        doc = {
            "$set": {
                "category": category,
                "is_active": True,
                "last_scraped_at": None
            },
            "$setOnInsert": {"created_at": datetime.now(timezone.utc)}
        }
        self.keywords_collection.update_one({"_id": keyword}, doc, upsert=True)
        print(f"   - Keyword '{keyword}' added/updated in category '{category}'.")

    def get_active_keywords(self) -> list:
        """Returns a list of all active keywords for the scraper."""
        cursor = self.keywords_collection.find({"is_active": True}, {"_id": 1})
        return [doc["_id"] for doc in cursor]

    def update_last_scraped_time(self, keyword: str):
        """Updates the last_scraped_at timestamp for a given keyword."""
        self.keywords_collection.update_one(
            {"_id": keyword},
            {"$set": {"last_scraped_at": datetime.now(timezone.utc)}}
        )
        print(f"   - Updated scrape time for '{keyword}'.")

    # --- Tweet CRUD Operations ---
    def save_tweets_batch(self, tweets: list):
        """Saves a batch of tweets efficiently, ignoring duplicates based on _id."""
        if not tweets:
            return 0
        operations = [
            UpdateOne({"_id": t.get("_id")}, {"$set": t}, upsert=True) for t in tweets
        ]
        try:
            result = self.tweets_collection.bulk_write(operations, ordered=False)
            print(f"   - Bulk write result: {result.upserted_count} new tweets saved.")
            return result.upserted_count
        except errors.BulkWriteError as bwe:
            print(f"❌ Bulk write error. Details: {bwe.details}")
            return 0

    def get_unprocessed_tweets(self, limit: int = 100) -> list:
        """Retrieves a batch of unprocessed tweets for the cleaning pipeline."""
        return list(self.tweets_collection.find({"is_processed": False}).limit(limit))

    def mark_tweets_as_processed(self, tweet_ids: list):
        """Marks a list of tweets as processed after cleaning."""
        if not tweet_ids:
            return
        result = self.tweets_collection.update_many(
            {"_id": {"$in": tweet_ids}},
            {"$set": {"is_processed": True}}
        )
        print(f"   - Marked {result.modified_count} tweets as processed.")
    
        # --- Utility: Load keywords from JSON file ---
        def load_keywords_from_json(self, json_path: str):
            """
            Loads keywords from a JSON file and inserts/updates them in the database.
            Each item should have '_id' (keyword) and 'category'.
            """
            import json
            with open(json_path, 'r', encoding='utf-8') as f:
                keywords = json.load(f)
            for item in keywords:
                keyword = item.get('_id')
                category = item.get('category', 'uncategorized')
                if keyword:
                    self.add_keyword(keyword, category)
            print(f"✅ Loaded {len(keywords)} keywords from {json_path}.")
