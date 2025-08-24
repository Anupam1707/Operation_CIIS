# storage.py

import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient, errors
from pymongo.operations import UpdateOne

class XBotDetectorDB:
    """
    Handles all MongoDB operations for the X Bot Detector project.
    This class is the dedicated 'engine' for database interactions.
    It does not read files; it only accepts and returns Python data.
    """
    def __init__(self):
        """Initializes the database connection using credentials from .env file or Codespaces secrets."""
        load_dotenv()
        mongo_uri = os.getenv("MONGO_URI")
        
        if not mongo_uri:
            raise ValueError("MONGO_URI not found. Please check your .env file or Codespaces secrets.")
            
        try:
            self.client = MongoClient(mongo_uri)
            self.client.admin.command('ping')
            self.db = self.client.x_bot_detector_db
            self.keywords_collection = self.db.keywords
            self.tweets_collection = self.db.tweets

            # --- Schema and Indexing ---
            # We will use 'keyword' as the unique identifier for keywords.
            self.keywords_collection.create_index("keyword", unique=True)
            # Tweets use their own ID from the API.
            self.tweets_collection.create_index("_id", unique=True)
            
            print("✅ MongoDB connection successful.")
        except errors.ConnectionFailure as e:
            print(f"❌ Could not connect to MongoDB: {e}")
            raise

    def close_connection(self):
        """Closes the MongoDB connection."""
        self.client.close()

    # --- Keyword Management Methods ---
    def add_keywords_bulk(self, keywords_data: list):
        """
        Efficiently adds or updates a list of keywords using a single bulk operation.
        This is the primary method for loading Srishti's data.
        
        Args:
            keywords_data (list): A list of dicts, each with 'keyword', 'category', 'description'.
        
        Returns:
            A dict with counts of added and updated keywords.
        """
        if not keywords_data:
            return {'added': 0, 'updated': 0}

        operations = []
        for kw_doc in keywords_data:
            op = UpdateOne(
                {"keyword": kw_doc['keyword']},
                {
                    "$set": {
                        "category": kw_doc.get('category', 'uncategorized'),
                        "description": kw_doc.get('description', ''),
                        "is_active": True
                    },
                    "$setOnInsert": {
                        "created_at": datetime.now(timezone.utc),
                        "last_scraped_at": None # Set to None on first insert
                    }
                },
                upsert=True
            )
            operations.append(op)
            
        try:
            result = self.keywords_collection.bulk_write(operations, ordered=False)
            return {'added': result.upserted_count, 'updated': result.modified_count}
        except errors.BulkWriteError as bwe:
            print(f"❌ Bulk write error during keyword update: {bwe.details}")
            return {'added': 0, 'updated': 0}

    def get_active_keywords(self) -> list:
        """Returns a list of all active keyword strings for Jayendra's scraper."""
        cursor = self.keywords_collection.find({"is_active": True}, {"keyword": 1, "_id": 0})
        return [doc["keyword"] for doc in cursor]

    # --- Tweet Management Methods ---
    # The original methods you provided are great, let's keep them.
    # Just renamed class from MongoHandler to XBotDetectorDB
    def save_tweets_batch(self, tweets: list):
        """Saves a batch of tweets efficiently, ignoring duplicates based on _id."""
        if not tweets: return 0
        operations = [UpdateOne({"_id": t.get("_id")}, {"$set": t}, upsert=True) for t in tweets]
        try:
            result = self.tweets_collection.bulk_write(operations, ordered=False)
            print(f"   - Bulk write result: {result.upserted_count} new tweets saved.")
            return result.upserted_count
        except errors.BulkWriteError as bwe:
            print(f"❌ Bulk write error saving tweets: {bwe.details}")
            return 0
    
    # ... Other tweet methods like get_unprocessed_tweets remain the same ...
