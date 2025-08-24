# storage.py

import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pymongo import MongoClient, errors, ASCENDING, DESCENDING
from pymongo.operations import UpdateOne, InsertOne
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XBotDetectorDB:
    """
    Handles all MongoDB operations for the X Bot Detector project.
    This class is the dedicated 'engine' for database interactions.
    It does not read files; it only accepts and returns Python data.
    """
    
    def __init__(self, database_name: str = "x_bot_detector_db"):
        """Initializes the database connection using credentials from .env file or Codespaces secrets."""
        load_dotenv()
        
        # Try multiple environment variable names for flexibility
        mongo_uri = (
            os.getenv("MONGO_URI") or 
            os.getenv("MONGODB_URI") or 
            os.getenv("MONGODB_CONNECTION_STRING")
        )
        
        if not mongo_uri:
            raise ValueError(
                "MongoDB connection string not found. Please set one of: "
                "MONGO_URI, MONGODB_URI, or MONGODB_CONNECTION_STRING in your .env file"
            )
            
        try:
            # Add connection timeout and server selection timeout
            self.client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                socketTimeoutMS=10000
            )
            
            # Test connection
            self.client.admin.command('ping')
            
            # Initialize database and collections
            self.db = self.client[database_name]
            self.keywords_collection = self.db.keywords
            self.tweets_collection = self.db.tweets
            self.bot_analysis_collection = self.db.bot_analysis
            
            # Create indexes for better performance
            self._create_indexes()
            
            logger.info("✅ MongoDB connection successful.")
            
        except errors.ConnectionFailure as e:
            logger.error(f"❌ Could not connect to MongoDB: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error during database initialization: {e}")
            raise

    def _create_indexes(self):
        """Creates necessary indexes for optimal performance."""
        try:
            # Keywords collection indexes
            self.keywords_collection.create_index("keyword", unique=True)
            self.keywords_collection.create_index("category")
            self.keywords_collection.create_index("is_active")
            self.keywords_collection.create_index("created_at")
            
            # Tweets collection indexes
            self.tweets_collection.create_index("tweet_id", unique=True)  # Use tweet_id, not _id
            self.tweets_collection.create_index("user_id")
            self.tweets_collection.create_index("created_at")
            self.tweets_collection.create_index("keyword_used")
            self.tweets_collection.create_index("processed")
            self.tweets_collection.create_index([("created_at", DESCENDING)])
            
            # Bot analysis collection indexes
            self.bot_analysis_collection.create_index("tweet_id", unique=True)
            self.bot_analysis_collection.create_index("user_id")
            self.bot_analysis_collection.create_index("bot_score")
            
            logger.info("📊 Database indexes created successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ Warning: Could not create some indexes: {e}")

    def close_connection(self):
        """Closes the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("🔌 MongoDB connection closed")

    # --- Keyword Management Methods ---
    
    def add_keywords_bulk(self, keywords_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Efficiently adds or updates a list of keywords using a single bulk operation.
        This is the primary method for loading Srishti's data.
        
        Args:
            keywords_data: A list of dicts, each with 'keyword', 'category', 'description', etc.
        
        Returns:
            A dict with counts of added and skipped keywords.
        """
        if not keywords_data:
            return {'added': 0, 'skipped': 0}

        operations = []
        valid_count = 0
        
        for kw_doc in keywords_data:
            # Validate required fields
            if not kw_doc.get('keyword') or not kw_doc.get('category'):
                logger.warning(f"Skipping invalid keyword data: {kw_doc}")
                continue
                
            # Normalize keyword (lowercase, strip whitespace)
            normalized_keyword = str(kw_doc['keyword']).lower().strip()
            
            if not normalized_keyword:
                logger.warning(f"Skipping empty keyword after normalization: {kw_doc}")
                continue
            
            op = UpdateOne(
                {"keyword": normalized_keyword},
                {
                    "$set": {
                        "category": str(kw_doc.get('category', 'uncategorized')).lower().strip(),
                        "description": str(kw_doc.get('description', '')).strip(),
                        "is_active": bool(kw_doc.get('is_active', True)),
                        "updated_at": datetime.now(timezone.utc)
                    },
                    "$setOnInsert": {
                        "created_at": datetime.now(timezone.utc),
                        "last_scraped_at": None,
                        "tweet_count": 0
                    }
                },
                upsert=True
            )
            operations.append(op)
            valid_count += 1
            
        if not operations:
            logger.warning("No valid keywords to process")
            return {'added': 0, 'skipped': len(keywords_data)}
            
        try:
            result = self.keywords_collection.bulk_write(operations, ordered=False)
            added_count = result.upserted_count
            skipped_count = len(keywords_data) - valid_count
            
            logger.info(f"📥 Keyword bulk operation completed: {added_count} added, {skipped_count} skipped")
            return {'added': added_count, 'skipped': skipped_count}
            
        except errors.BulkWriteError as bwe:
            logger.error(f"❌ Bulk write error during keyword update: {bwe.details}")
            return {'added': 0, 'skipped': len(keywords_data)}
        except Exception as e:
            logger.error(f"❌ Unexpected error during keyword bulk operation: {e}")
            return {'added': 0, 'skipped': len(keywords_data)}

    def get_active_keywords(self) -> List[str]:
        """Returns a list of all active keyword strings for Jayendra's scraper."""
        try:
            cursor = self.keywords_collection.find(
                {"is_active": True}, 
                {"keyword": 1, "_id": 0}
            )
            keywords = [doc["keyword"] for doc in cursor]
            logger.info(f"📋 Retrieved {len(keywords)} active keywords")
            return keywords
        except Exception as e:
            logger.error(f"❌ Error retrieving active keywords: {e}")
            return []

    def get_keywords_by_category(self, category: str = None, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get keywords filtered by category and/or active status."""
        try:
            query = {}
            if category:
                query["category"] = category.lower().strip()
            if active_only:
                query["is_active"] = True
                
            cursor = self.keywords_collection.find(query)
            keywords = list(cursor)
            logger.info(f"📋 Retrieved {len(keywords)} keywords for category: {category}")
            return keywords
        except Exception as e:
            logger.error(f"❌ Error retrieving keywords by category: {e}")
            return []

    def update_keyword_stats(self, keyword: str, increment_count: int = 1):
        """Update keyword usage statistics."""
        try:
            result = self.keywords_collection.update_one(
                {"keyword": keyword.lower().strip()},
                {
                    "$inc": {"tweet_count": increment_count},
                    "$set": {"last_scraped_at": datetime.now(timezone.utc)}
                }
            )
            if result.modified_count > 0:
                logger.debug(f"Updated stats for keyword: {keyword}")
        except Exception as e:
            logger.error(f"❌ Error updating keyword stats for {keyword}: {e}")

    # --- Tweet Management Methods ---
    
    def save_tweets_batch(self, tweets: List[Dict[str, Any]]) -> int:
        """
        Saves a batch of tweets efficiently, ignoring duplicates based on tweet_id.
        
        Args:
            tweets: List of tweet dictionaries
            
        Returns:
            Number of tweets successfully saved
        """
        if not tweets:
            return 0
            
        operations = []
        valid_count = 0
        
        for tweet in tweets:
            # Validate required fields
            if not tweet.get('tweet_id'):
                logger.warning(f"Skipping tweet without tweet_id: {tweet}")
                continue
                
            # Add metadata
            tweet_doc = {
                **tweet,
                "scraped_at": datetime.now(timezone.utc),
                "processed": False,
                "bot_analyzed": False
            }
            
            op = UpdateOne(
                {"tweet_id": tweet["tweet_id"]},
                {"$set": tweet_doc},
                upsert=True
            )
            operations.append(op)
            valid_count += 1
        
        if not operations:
            logger.warning("No valid tweets to save")
            return 0
            
        try:
            result = self.tweets_collection.bulk_write(operations, ordered=False)
            saved_count = result.upserted_count
            
            # Update keyword stats for tweets with keyword_used
            for tweet in tweets:
                if tweet.get('keyword_used'):
                    self.update_keyword_stats(tweet['keyword_used'])
            
            logger.info(f"💾 Saved {saved_count} new tweets out of {valid_count} valid tweets")
            return saved_count
            
        except errors.BulkWriteError as bwe:
            logger.error(f"❌ Bulk write error saving tweets: {bwe.details}")
            return 0
        except Exception as e:
            logger.error(f"❌ Unexpected error saving tweets: {e}")
            return 0

    def get_unprocessed_tweets(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Returns tweets that haven't been processed for NLP analysis."""
        try:
            cursor = self.tweets_collection.find(
                {"processed": False}
            ).limit(limit).sort("created_at", ASCENDING)
            
            tweets = list(cursor)
            logger.info(f"📤 Retrieved {len(tweets)} unprocessed tweets")
            return tweets
        except Exception as e:
            logger.error(f"❌ Error retrieving unprocessed tweets: {e}")
            return []

    def mark_tweets_processed(self, tweet_ids: List[str]) -> int:
        """Mark tweets as processed after cleaning."""
        try:
            result = self.tweets_collection.update_many(
                {"tweet_id": {"$in": tweet_ids}},
                {
                    "$set": {
                        "processed": True,
                        "processed_at": datetime.now(timezone.utc)
                    }
                }
            )
            logger.info(f"✅ Marked {result.modified_count} tweets as processed")
            return result.modified_count
        except Exception as e:
            logger.error(f"❌ Error marking tweets as processed: {e}")
            return 0

    def get_tweets_by_keyword(self, keyword: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get tweets associated with a specific keyword."""
        try:
            cursor = self.tweets_collection.find(
                {"keyword_used": keyword.lower().strip()}
            ).limit(limit).sort("created_at", DESCENDING)
            
            tweets = list(cursor)
            logger.info(f"📊 Retrieved {len(tweets)} tweets for keyword: {keyword}")
            return tweets
        except Exception as e:
            logger.error(f"❌ Error retrieving tweets for keyword {keyword}: {e}")
            return []

    # --- Bot Analysis Methods ---
    
    def save_bot_analysis(self, analysis_data: Dict[str, Any]) -> bool:
        """Save bot analysis results for a tweet."""
        try:
            if not analysis_data.get('tweet_id'):
                logger.error("Cannot save bot analysis without tweet_id")
                return False
                
            analysis_doc = {
                **analysis_data,
                "analyzed_at": datetime.now(timezone.utc)
            }
            
            # Use upsert to update if exists
            result = self.bot_analysis_collection.update_one(
                {"tweet_id": analysis_data["tweet_id"]},
                {"$set": analysis_doc},
                upsert=True
            )
            
            # Mark tweet as analyzed
            self.tweets_collection.update_one(
                {"tweet_id": analysis_data["tweet_id"]},
                {"$set": {"bot_analyzed": True}}
            )
            
            logger.info(f"🤖 Saved bot analysis for tweet: {analysis_data['tweet_id']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving bot analysis: {e}")
            return False

    # --- Utility Methods ---
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            # Keywords stats
            keywords_total = self.keywords_collection.count_documents({})
            keywords_active = self.keywords_collection.count_documents({"is_active": True})
            
            # Category breakdown
            category_pipeline = [
                {"$group": {"_id": "$category", "count": {"$sum": 1}}}
            ]
            categories = list(self.keywords_collection.aggregate(category_pipeline))
            
            # Tweets stats
            tweets_total = self.tweets_collection.count_documents({})
            tweets_processed = self.tweets_collection.count_documents({"processed": True})
            tweets_bot_analyzed = self.tweets_collection.count_documents({"bot_analyzed": True})
            
            # Recent tweets (last 24 hours)
            yesterday = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            tweets_recent = self.tweets_collection.count_documents({
                "created_at": {"$gte": yesterday}
            })
            
            # Bot analysis stats
            bot_analysis_total = self.bot_analysis_collection.count_documents({})
            
            stats = {
                "keywords": {
                    "total": keywords_total,
                    "active": keywords_active,
                    "by_category": categories
                },
                "tweets": {
                    "total": tweets_total,
                    "processed": tweets_processed,
                    "bot_analyzed": tweets_bot_analyzed,
                    "last_24h": tweets_recent
                },
                "bot_analysis": {
                    "total": bot_analysis_total
                },
                "database_name": self.db.name,
                "collections": self.db.list_collection_names()
            }
            
            logger.info("📈 Generated database statistics")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error generating database stats: {e}")
            return {
                "keywords": {"total": 0, "active": 0, "by_category": []},
                "tweets": {"total": 0, "processed": 0, "bot_analyzed": 0, "last_24h": 0},
                "bot_analysis": {"total": 0},
                "error": str(e)
            }

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the database connection."""
        try:
            # Test connection
            self.client.admin.command('ping')
            
            # Test basic operations
            test_result = self.keywords_collection.find_one({}, {"_id": 1})
            
            return {
                "status": "healthy",
                "connection": "active",
                "database": self.db.name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_connection()


# --- Convenience Functions ---

def test_connection():
    """Test database connection with sample operations."""
    try:
        with XBotDetectorDB() as db:
            print("🧪 Testing database connection...")
            
            # Health check
            health = db.health_check()
            print(f"Health check: {health['status']}")
            
            # Get stats
            stats = db.get_database_stats()
            print(f"Total keywords: {stats['keywords']['total']}")
            print(f"Total tweets: {stats['tweets']['total']}")
            
            print("✅ Database test completed successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


if __name__ == "__main__":
    # Run test if script is executed directly
    test_connection()
