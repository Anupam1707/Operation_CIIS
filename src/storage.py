import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pymongo import MongoClient, errors, ASCENDING, DESCENDING
from pymongo.operations import UpdateOne
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoHandler:
    """
    Handles all MongoDB operations for the project.
    """
    
    def __init__(self, database_name: str = "operation_ciis_db"):
        """Initializes the database connection."""
        load_dotenv()
        
        mongo_uri = os.getenv("MONGO_URI")
        
        if not mongo_uri:
            raise ValueError("MONGO_URI not found in .env file")
            
        try:
            self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client[database_name]
            self.keywords_collection = self.db.keywords
            self.tweets_collection = self.db.tweets
            self._create_indexes()
            logger.info("MongoDB connection successful.")
            
        except errors.ConnectionFailure as e:
            logger.error(f"Could not connect to MongoDB: {e}")
            raise

    def _create_indexes(self):
        """Creates necessary indexes."""
        self.keywords_collection.create_index("keyword", unique=True)
        self.tweets_collection.create_index("tweet_id", unique=True)
        self.tweets_collection.create_index("user_id")
        self.tweets_collection.create_index([("created_at", DESCENDING)])

    def get_active_keywords(self) -> List[str]:
        """Returns a list of all active keywords."""
        try:
            cursor = self.keywords_collection.find({"is_active": True}, {"keyword": 1, "_id": 0})
            return [doc["keyword"] for doc in cursor]
        except Exception as e:
            logger.error(f"Error retrieving active keywords: {e}")
            return []

    def save_tweets_batch(self, tweets: List[Dict[str, Any]]) -> int:
        """Saves a batch of tweets, ignoring duplicates."""
        if not tweets:
            return 0
            
        operations = [
            UpdateOne({"_id": tweet["_id"]}, {"$set": tweet}, upsert=True)
            for tweet in tweets
        ]
        
        try:
            result = self.tweets_collection.bulk_write(operations, ordered=False)
            return result.upserted_count
        except errors.BulkWriteError as bwe:
            logger.error(f"Bulk write error saving tweets: {bwe.details}")
            return 0

    def update_last_scraped_time(self, keyword: str):
        """Updates the last scraped time for a keyword."""
        self.keywords_collection.update_one(
            {"keyword": keyword},
            {"$set": {"last_scraped_at": datetime.now(timezone.utc)}}
        )

if __name__ == "__main__":
    try:
        handler = MongoHandler()
        print("Successfully connected to MongoDB.")
        # Example: Add a keyword
        # handler.keywords_collection.update_one({"keyword": "india"}, {"$set": {"is_active": True}}, upsert=True)
        # print(handler.get_active_keywords())
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
