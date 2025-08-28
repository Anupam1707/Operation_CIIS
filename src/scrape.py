import os
import sys
import time
import json
import logging
import re
import signal
import random
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_setup import create_headers, BEARER_TOKEN
from storage import MongoHandler

# Setup paths and logging
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

LOG_PATH = os.path.join(DATA_DIR, "scraper.log")
SUMMARY_PATH_DEFAULT = os.path.join(DATA_DIR, "scraping_summary.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScrapingConfig:
    """Configuration constants for the scraper"""
    DEFAULT_MAX_TWEETS = 500
    MAX_TWEETS_LIMIT = 1000
    TWEETS_PER_REQUEST = 100
    DELAY_BETWEEN_REQUESTS = 1
    DELAY_BETWEEN_KEYWORDS = 2
    REQUEST_TIMEOUT = 30
    
    # Retry configuration
    MAX_RETRIES = 5
    BASE_BACKOFF = 1.0
    MAX_BACKOFF = 60.0
    JITTER_RANGE = 0.1
    
    TWEET_FIELDS = [
        "author_id",
        "created_at", 
        "public_metrics",
        "lang",
        "conversation_id",
        "possibly_sensitive",
        "context_annotations",
        "referenced_tweets"
    ]
    
    DEFAULT_FILTERS = {
        "include_retweets": False,
        "include_quotes": False,
        "include_replies": False,
        "language": "en",
    }


class GracefulKiller:
    """Handle graceful shutdown signals"""
    
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self._exit_gracefully)
        signal.signal(signal.SIGTERM, self._exit_gracefully)

    def _exit_gracefully(self, signum, frame):
        logger.info(f"Received shutdown signal {signum}. Gracefully shutting down...")
        self.kill_now = True


class ScrapingMetrics:
    """Track scraping performance metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.api_calls = 0
        self.tweets_processed = 0
        self.tweets_saved = 0
        self.tweets_skipped = 0
        self.rate_limit_hits = 0
        self.errors = 0
        self.retries = 0
        self.keyword_stats = {}
        self.checkpoints_saved = 0
    
    def record_api_call(self):
        self.api_calls += 1
    
    def record_tweets_processed(self, count: int):
        self.tweets_processed += count
    
    def record_tweets_saved(self, count: int):
        self.tweets_saved += count
    
    def record_tweets_skipped(self, count: int):
        self.tweets_skipped += count
    
    def record_rate_limit_hit(self):
        self.rate_limit_hits += 1
    
    def record_error(self):
        self.errors += 1
    
    def record_retry(self):
        self.retries += 1
    
    def record_checkpoint(self):
        self.checkpoints_saved += 1
    
    def record_keyword_result(self, keyword: str, saved_count: int, api_calls: int):
        self.keyword_stats[keyword] = {
            "saved": saved_count,
            "api_calls": api_calls,
            "efficiency": saved_count / api_calls if api_calls > 0 else 0
        }
    
    def get_summary(self) -> Dict:
        duration = time.time() - self.start_time
        return {
            "duration_seconds": round(duration, 2),
            "api_calls": self.api_calls,
            "tweets_processed": self.tweets_processed,
            "tweets_saved": self.tweets_saved,
            "tweets_skipped": self.tweets_skipped,
            "save_rate": self.tweets_saved / self.tweets_processed if self.tweets_processed > 0 else 0,
            "rate_limit_hits": self.rate_limit_hits,
            "errors": self.errors,
            "retries": self.retries,
            "tweets_per_second": self.tweets_saved / duration if duration > 0 else 0,
            "checkpoints_saved": self.checkpoints_saved,
            "keyword_stats": self.keyword_stats
        }


class TwitterScraper:
    """
    Enhanced Twitter/X scraper with exponential backoff, graceful shutdown,
    checkpointing, and safe retry logic.
    """

    def __init__(self, mongo_handler: MongoHandler):
        self.mongo_handler = mongo_handler
        self.metrics = ScrapingMetrics()
        self.killer = GracefulKiller()
        
        api_base = os.getenv("TWITTER_API_BASE", "https://api.x.com/2")
        self.base_url = f"{api_base.rstrip('/')}/tweets/search/recent"
        self.headers = create_headers()

        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            connect=3,
            read=2,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.checkpoints_collection = self.mongo_handler.db.scraper_checkpoints
        self.checkpoints_collection.create_index("keyword", unique=True)

    def _update_since_id(self, current_max: Optional[str], new_id: str) -> str:
        """Update since_id to track highest tweet ID seen"""
        try:
            return str(max(int(current_max or 0), int(new_id)))
        except (ValueError, TypeError):
            return current_max or new_id

    def _exponential_backoff_with_jitter(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter"""
        backoff = min(
            ScrapingConfig.BASE_BACKOFF * (2 ** attempt),
            ScrapingConfig.MAX_BACKOFF
        )
        jitter = backoff * ScrapingConfig.JITTER_RANGE * random.random()
        return backoff + jitter

    def _save_checkpoint(self, keyword: str, next_token: Optional[str], 
                        tweets_saved: int, last_tweet_id: Optional[str] = None,
                        since_id: Optional[str] = None):
        """Save scraping progress checkpoint"""
        checkpoint = {
            "keyword": keyword,
            "next_token": next_token,
            "tweets_saved": tweets_saved,
            "last_tweet_id": last_tweet_id,
            "since_id": since_id,
            "updated_at": datetime.now(timezone.utc)
        }
        
        self.checkpoints_collection.update_one(
            {"keyword": keyword},
            {"$set": checkpoint},
            upsert=True
        )
        self.metrics.record_checkpoint()
        logger.debug(f"Checkpoint saved for '{keyword}' - saved: {tweets_saved}")

    def _load_checkpoint(self, keyword: str) -> Optional[Dict]:
        """Load existing checkpoint for keyword"""
        checkpoint = self.checkpoints_collection.find_one({"keyword": keyword})
        if checkpoint:
            logger.info(f"Resuming from checkpoint for '{keyword}' - "
                       f"previously saved: {checkpoint.get('tweets_saved', 0)}")
        return checkpoint

    def _clear_checkpoint(self, keyword: str):
        """Clear checkpoint after successful completion"""
        self.checkpoints_collection.delete_one({"keyword": keyword})
        logger.debug(f"Checkpoint cleared for '{keyword}'")

    def build_search_query(self, keyword: str, **filters) -> str:
        """Build flexible search query with configurable filters"""
        config = {**ScrapingConfig.DEFAULT_FILTERS, **filters}
        query_parts = [keyword.strip()]
        
        if not config["include_retweets"]:
            query_parts.append("-is:retweet")
        
        if not config["include_quotes"]:
            query_parts.append("-is:quote")
        
        if not config["include_replies"]:
            query_parts.append("-is:reply")
        
        if config["language"]:
            query_parts.append(f"lang:{config['language']}")
        
        return " ".join(query_parts)

    def _validate_tweet_data(self, tweet: Dict) -> bool:
        """Validate tweet data quality"""
        if not all(key in tweet for key in ["id", "text"]):
            return False
        
        text = tweet.get("text", "").strip()
        if len(text) < 15:
            return False
        
        # Advanced spam detection could be added here
        
        return True

    def _make_request(self, params: Dict) -> Optional[Dict]:
        """
        Make API request with retry logic handled by requests.Session.
        Returns response data or None.
        """
        if self.killer.kill_now:
            logger.info("Shutdown requested, stopping request")
            return None

        try:
            self.metrics.record_api_call()
            resp = self.session.get(
                self.base_url,
                headers=self.headers,
                params=params,
                timeout=ScrapingConfig.REQUEST_TIMEOUT
            )
            resp.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            return resp.json()

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error: {e.response.status_code} {e.response.text[:500]}")
            if e.response.status_code == 429:
                self.metrics.record_rate_limit_hit()
                reset_time = int(e.response.headers.get("x-rate-limit-reset", time.time() + 60))
                wait = max(0, reset_time - int(time.time())) + 2
                logger.warning(f"Rate limited. Waiting {wait}s.")
                time.sleep(wait)
            self.metrics.record_error()
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            self.metrics.record_error()
            return None

    def _parse_tweet_data(self, tweet: Dict, keyword: str) -> Dict:
        """Parse tweet data into standardized format"""
        pm = tweet.get("public_metrics", {})
        now_iso = datetime.now(timezone.utc).isoformat()

        return {
            "_id": tweet["id"],
            "text": tweet["text"],
            "user_id": tweet.get("author_id"),
            "timestamp": tweet.get("created_at"),
            "lang": tweet.get("lang", "unknown"),
            "keyword_scraped": keyword,
            "retweet_count": pm.get("retweet_count", 0),
            "like_count": pm.get("like_count", 0),
            "reply_count": pm.get("reply_count", 0),
            "quote_count": pm.get("quote_count", 0),
            "is_processed": False,
            "scraped_at": now_iso,
            **{k: tweet.get(k) for k in ["conversation_id", "possibly_sensitive", "context_annotations"]}
        }

    def fetch_tweets_for_keyword(self, keyword: str, max_tweets: int) -> int:
        """Fetch tweets for a single keyword."""
        max_tweets = min(int(max_tweets), ScrapingConfig.MAX_TWEETS_LIMIT)
        checkpoint = self._load_checkpoint(keyword)
        total_saved = checkpoint.get("tweets_saved", 0) if checkpoint else 0
        next_token = checkpoint.get("next_token") if checkpoint else None
        since_id = checkpoint.get("since_id") if checkpoint else None

        logger.info(f"Starting keyword '{keyword}' - Target: {max_tweets}, Saved: {total_saved}")

        while total_saved < max_tweets:
            if self.killer.kill_now: break

            params = {
                "query": self.build_search_query(keyword),
                "max_results": min(ScrapingConfig.TWEETS_PER_REQUEST, max_tweets - total_saved),
                "tweet.fields": ",".join(ScrapingConfig.TWEET_FIELDS)
            }
            if since_id: params["since_id"] = since_id
            elif next_token: params["next_token"] = next_token

            response_data = self._make_request(params)
            if not response_data or "data" not in response_data:
                break

            tweets = response_data.get("data", [])
            valid_tweets = [self._parse_tweet_data(t, keyword) for t in tweets if self._validate_tweet_data(t)]
            
            if valid_tweets:
                saved_count = self.mongo_handler.save_tweets_batch(valid_tweets)
                total_saved += saved_count
                logger.info(f"Saved {saved_count} new tweets for '{keyword}'. Total: {total_saved}")
                since_id = self._update_since_id(since_id, valid_tweets[0]['_id'])

            self.metrics.record_tweets_processed(len(tweets))
            self.metrics.record_tweets_saved(len(valid_tweets))

            next_token = response_data.get("meta", {}).get("next_token")
            self._save_checkpoint(keyword, next_token, total_saved, since_id=since_id)

            if not next_token:
                logger.info(f"No more pages for '{keyword}'.")
                break
            
            time.sleep(ScrapingConfig.DELAY_BETWEEN_REQUESTS)

        if total_saved >= max_tweets or not next_token:
            self._clear_checkpoint(keyword)
        
        self.metrics.record_keyword_result(keyword, total_saved, self.metrics.api_calls)
        return total_saved

    def scrape_all_keywords(self, max_tweets_per_keyword: int) -> Dict[str, int]:
        """Scrape all active keywords from the database."""
        logger.info("Starting bulk scraping process...")
        keywords = self.mongo_handler.get_active_keywords()
        if not keywords:
            logger.warning("No active keywords found.")
            return {}

        results = {}
        for i, keyword in enumerate(keywords, 1):
            if self.killer.kill_now: break
            logger.info(f"--- Processing keyword {i}/{len(keywords)}: '{keyword}' ---")
            results[keyword] = self.fetch_tweets_for_keyword(keyword, max_tweets_per_keyword)
            self.mongo_handler.update_last_scraped_time(keyword)
            if i < len(keywords): time.sleep(ScrapingConfig.DELAY_BETWEEN_KEYWORDS)

        logger.info("Scraping finished.")
        return results

def export_scraping_summary(results: Dict[str, int], metrics: ScrapingMetrics, output_path: str):
    summary = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "total_keywords": len(results),
        "total_tweets_saved": sum(results.values()),
        "results_by_keyword": results,
        "performance": metrics.get_summary()
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Scraping summary saved to {output_path}")

if __name__ == "__main__":
    if not BEARER_TOKEN: raise ValueError("BEARER_TOKEN not set in .env file")
    
    mongo_handler = MongoHandler()
    scraper = TwitterScraper(mongo_handler)
    
    results = scraper.scrape_all_keywords(ScrapingConfig.DEFAULT_MAX_TWEETS)
    
    export_scraping_summary(results, scraper.metrics, SUMMARY_PATH_DEFAULT)
    
    print(json.dumps(scraper.metrics.get_summary(), indent=2))
