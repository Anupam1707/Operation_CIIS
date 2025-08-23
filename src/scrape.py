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
        "min_engagement": False  # Default to False to avoid 400/422 errors
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
        # Initialize graceful shutdown handler
        self.killer = GracefulKiller()
        
        # Windows signal handling fallback
        try:
            import signal
            signal.signal(signal.SIGTERM, self.killer._exit_gracefully)
        except (ImportError, AttributeError):
            # Windows may not have SIGTERM, wrap gracefully
            logger.info("SIGTERM not available, using basic shutdown handling")
        
        # API setup
        api_base = os.getenv("TWITTER_API_BASE", "https://api.x.com/2")
        self.base_url = f"{api_base.rstrip('/')}/tweets/search/recent"
        self.headers = create_headers()

        # Setup session with basic retry for connection issues only
        self.session = requests.Session()
        retry_strategy = Retry(
            total=2,  # Keep low, we handle retries manually
            connect=2,
            read=1,
            status=0,  # Don't auto-retry status codes
            backoff_factor=0.3,
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Collections for checkpointing
        self.checkpoints_collection = self.mongo_handler.db.scraper_checkpoints
        self.checkpoints_collection.create_index("keyword", unique=True)

        # Test API connectivity
        self._test_api_connection()

    def _test_api_connection(self):
        """Test API connectivity during initialization"""
        test_params = {"query": "test", "max_results": 1}
        try:
            response = self.session.get(
                self.base_url,
                headers=self.headers,
                params=test_params,
                timeout=10
            )
            if response.status_code not in [200, 429]:
                raise ConnectionError(f"API test failed: {response.status_code}")
            logger.info("✅ API connectivity test passed")
        except Exception as e:
            raise ConnectionError(f"Unable to connect to Twitter API: {e}")

    def _update_since_id(self, current_max: Optional[str], new_id: str) -> str:
        """Update since_id to track highest tweet ID seen"""
        try:
            return str(max(int(current_max or 0), int(new_id)))
        except Exception:
            return current_max or new_id

    def _exponential_backoff_with_jitter(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter"""
        backoff = min(
            ScrapingConfig.BASE_BACKOFF * (2 ** attempt),
            ScrapingConfig.MAX_BACKOFF
        )
        # Add jitter to avoid thundering herd
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

    def build_search_query(self, keyword: str, lite: bool = False, **filters) -> str:
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
        
        # Only add engagement filter if NOT lite mode and min_engagement is True
        if not lite and config["min_engagement"]:
            query_parts.append("(has:media OR has:links OR min_faves:1)")
        
        return " ".join(query_parts)

    def _validate_tweet_data(self, tweet: Dict) -> bool:
        """Validate tweet data quality"""
        # Essential fields check
        if not all(key in tweet for key in ["id", "text"]):
            return False
        
        text = tweet.get("text", "").strip()
        
        # Skip very short tweets
        if len(text) < 10:
            return False
        
        # Skip tweets that are mostly URLs/mentions
        url_count = len(re.findall(r'http\S+', text))
        mention_count = len(re.findall(r'@\w+', text))
        word_count = len(text.split())
        
        if word_count > 0 and (url_count + mention_count) / word_count > 0.7:
            return False
        
        # Skip spam indicators
        spam_indicators = [
            text.upper() == text and len(text) > 20,  # ALL CAPS
            text.count('!') > 5,  # Too many exclamations
            len(set(text.split())) / len(text.split()) < 0.3 if text.split() else False  # Too repetitive
        ]
        
        return not any(spam_indicators)

    def _make_request(self, params: Dict, keyword: str = "", tried_lite: bool = False) -> Tuple[Optional[Dict], bool]:
        """
        Make API request with safe retry logic and exponential backoff.
        Returns: (response_data, should_continue)
        """
        for attempt in range(ScrapingConfig.MAX_RETRIES):
            # Check for shutdown signal
            if self.killer.kill_now:
                logger.info("Shutdown requested, stopping request")
                return None, False
            
            try:
                self.metrics.record_api_call()
                
                resp = self.session.get(
                    self.base_url,
                    headers=self.headers,
                    params=params,
                    timeout=ScrapingConfig.REQUEST_TIMEOUT
                )

                # Success case
                if resp.status_code == 200:
                    # Check rate limit status for proactive handling
                    remaining = resp.headers.get("x-rate-limit-remaining")
                    reset = resp.headers.get("x-rate-limit-reset")
                    
                    if remaining and int(remaining) < 5:  # Proactive rate limit handling
                        if reset:
                            wait = max(0, int(reset) - int(datetime.now(timezone.utc).timestamp())) + 2
                            logger.info(f"Low rate limit remaining ({remaining}). "
                                      f"Sleeping {wait}s until reset.")
                            self.metrics.record_rate_limit_hit()
                            time.sleep(wait)
                    
                    return resp.json(), True

                # Rate limit hit
                elif resp.status_code == 429:
                    reset = resp.headers.get("x-rate-limit-reset")
                    wait = 60 if not reset else max(
                        0, int(reset) - int(datetime.now(timezone.utc).timestamp())
                    ) + 2
                    
                    logger.warning(f"Rate limited. Waiting {wait}s before retry {attempt + 1}")
                    self.metrics.record_rate_limit_hit()
                    time.sleep(wait)
                    continue

                # Client errors - don't retry most, but try lite mode for 400/422
                elif resp.status_code in (400, 422):
                    logger.warning(f"{resp.status_code} for '{keyword}': {resp.text[:500]}")
                    if not tried_lite and "query" in params:
                        # Switch to lite query (drop engagement filters)
                        params["query"] = self.build_search_query(keyword, lite=True)
                        self.metrics.record_retry()
                        continue
                    
                    self.metrics.record_error()
                    return None, True  # Continue with next keyword/request

                elif resp.status_code == 401:
                    logger.error("Authentication failed. Check bearer token.")
                    raise ValueError("Invalid or expired bearer token")
                
                elif resp.status_code == 403:
                    logger.error(f"Forbidden access for '{keyword}': {resp.text}")
                    self.metrics.record_error()
                    return None, True
                
                # Server errors - retry with backoff
                elif resp.status_code >= 500:
                    if attempt < ScrapingConfig.MAX_RETRIES - 1:
                        backoff = self._exponential_backoff_with_jitter(attempt)
                        logger.warning(f"Server error {resp.status_code}. "
                                     f"Retrying in {backoff:.1f}s (attempt {attempt + 1})")
                        self.metrics.record_retry()
                        time.sleep(backoff)
                        continue
                    else:
                        logger.error(f"Max retries exceeded for server error {resp.status_code}")
                        self.metrics.record_error()
                        return None, True

                # Other status codes
                else:
                    logger.error(f"Unexpected status code {resp.status_code}: {resp.text}")
                    self.metrics.record_error()
                    return None, True

            except requests.exceptions.Timeout:
                if attempt < ScrapingConfig.MAX_RETRIES - 1:
                    backoff = self._exponential_backoff_with_jitter(attempt)
                    logger.warning(f"Request timeout. Retrying in {backoff:.1f}s")
                    self.metrics.record_retry()
                    time.sleep(backoff)
                    continue
                else:
                    logger.error("Max retries exceeded for timeout")
                    self.metrics.record_error()
                    return None, True

            except requests.exceptions.RequestException as e:
                if attempt < ScrapingConfig.MAX_RETRIES - 1:
                    backoff = self._exponential_backoff_with_jitter(attempt)
                    logger.warning(f"Request exception: {e}. Retrying in {backoff:.1f}s")
                    self.metrics.record_retry()
                    time.sleep(backoff)
                    continue
                else:
                    logger.error(f"Max retries exceeded for request exception: {e}")
                    self.metrics.record_error()
                    return None, True

        # If we get here, all retries failed
        logger.error(f"All {ScrapingConfig.MAX_RETRIES} attempts failed")
        self.metrics.record_error()
        return None, True

    def _parse_tweet_data(self, tweet: Dict, keyword: str) -> Dict:
        """Parse tweet data into standardized format"""
        pm = tweet.get("public_metrics", {})
        tweet_created_at = tweet.get("created_at")
        now_iso = datetime.now(timezone.utc).isoformat()

        doc = {
            "_id": tweet["id"],
            "text": tweet["text"],
            "user_id": tweet.get("author_id"),
            "author_id": tweet.get("author_id"),
            "timestamp": tweet_created_at,
            "created_at": tweet_created_at,
            "lang": tweet.get("lang", "unknown"),
            "keyword_scraped": keyword,
            "retweet_count": pm.get("retweet_count", 0),
            "like_count": pm.get("like_count", 0),
            "reply_count": pm.get("reply_count", 0),
            "quote_count": pm.get("quote_count", 0),
            "is_processed": False,
            "scraped_at": now_iso
        }

        # Optional fields
        for field in ["conversation_id", "possibly_sensitive", "context_annotations"]:
            if field in tweet:
                doc[field] = tweet[field]

        return doc

    def fetch_tweets_for_keyword(self, keyword: str, max_tweets: int = None) -> int:
        """Fetch tweets for keyword with checkpointing and graceful shutdown"""
        if max_tweets is None:
            max_tweets = ScrapingConfig.DEFAULT_MAX_TWEETS
        
        max_tweets = max(1, min(int(max_tweets), ScrapingConfig.MAX_TWEETS_LIMIT))
        keyword_norm = keyword.strip()
        
        # Load checkpoint if exists
        checkpoint = self._load_checkpoint(keyword_norm)
        total_saved = checkpoint.get("tweets_saved", 0) if checkpoint else 0
        next_token = checkpoint.get("next_token") if checkpoint else None
        since_id = checkpoint.get("since_id") if checkpoint else None
        max_seen_id = self._update_since_id(since_id, "0") if since_id else None
        
        logger.info(f"Starting keyword '{keyword_norm}' - target: {max_tweets}, "
                   f"already saved: {total_saved}")

        total_processed = 0
        total_skipped = 0
        keyword_api_calls = 0
        tweets_per_request = min(ScrapingConfig.TWEETS_PER_REQUEST, max_tweets)
        last_tweet_id = None

        while total_saved < max_tweets:
            # Check for graceful shutdown
            if self.killer.kill_now:
                logger.info(f"Shutdown requested. Saving checkpoint for '{keyword_norm}'")
                self._save_checkpoint(keyword_norm, next_token, total_saved, last_tweet_id)
                return total_saved

            remaining_tweets = max_tweets - total_saved
            current_request_size = min(tweets_per_request, remaining_tweets)

            params = {
                "query": self.build_search_query(keyword_norm),
                "max_results": max(10, min(current_request_size, 100)),  # Clamp to [10, 100]
                "tweet.fields": ",".join(ScrapingConfig.TWEET_FIELDS)
            }

            # Prefer since_id to avoid stale pagination
            if since_id:
                params["since_id"] = since_id
            elif next_token:
                params["next_token"] = next_token

            logger.info(f"Fetching {current_request_size} tweets (saved: {total_saved})")
            
            response_data, should_continue = self._make_request(params, keyword_norm)
            keyword_api_calls += 1

            if not should_continue:
                # Shutdown or critical error
                self._save_checkpoint(keyword_norm, next_token, total_saved, last_tweet_id)
                return total_saved

            if not response_data:
                logger.error(f"No response data for '{keyword_norm}'")
                break

            tweets_data = response_data.get("data", [])
            if not tweets_data:
                logger.info(f"No more tweets for '{keyword_norm}'")
                break

            # Validate and parse tweets, tracking keywords_matched for duplicates
            valid_tweets = []
            for tweet in tweets_data:
                total_processed += 1
                if self._validate_tweet_data(tweet):
                    parsed_tweet = self._parse_tweet_data(tweet, keyword_norm)
                    
                    # Track highest ID seen for since_id checkpointing
                    max_seen_id = self._update_since_id(max_seen_id, tweet["id"])
                    
                    # Handle duplicate tweets across keywords by storing array
                    existing_tweet = self.mongo_handler.tweets_collection.find_one({"_id": tweet["id"]})
                    if existing_tweet:
                        # Add to keywords_matched array instead of overwriting
                        keywords_matched = existing_tweet.get("keywords_matched", [existing_tweet.get("keyword_scraped", "")])
                        if keyword_norm not in keywords_matched:
                            keywords_matched.append(keyword_norm)
                            self.mongo_handler.tweets_collection.update_one(
                                {"_id": tweet["id"]},
                                {"$set": {"keywords_matched": keywords_matched}}
                            )
                    else:
                        parsed_tweet["keywords_matched"] = [keyword_norm]
                        valid_tweets.append(parsed_tweet)
                    
                    last_tweet_id = tweet["id"]
                else:
                    total_skipped += 1

            # Save valid tweets
            if valid_tweets:
                saved_count = self.mongo_handler.save_tweets_batch(valid_tweets)
                total_saved += saved_count
                
                logger.info(f"Processed: {len(tweets_data)}, Valid: {len(valid_tweets)}, "
                           f"Saved: {saved_count}, Total: {total_saved}")

            # Update pagination
            meta = response_data.get("meta", {})
            next_token = meta.get("next_token")
            
            # Save checkpoint every 100 tweets or when no next_token
            if total_saved % 100 == 0 or not next_token:
                self._save_checkpoint(keyword_norm, next_token, total_saved, last_tweet_id, max_seen_id)

            if not next_token:
                logger.info(f"No more pages for '{keyword_norm}'")
                break

            time.sleep(ScrapingConfig.DELAY_BETWEEN_REQUESTS)

        # Clear checkpoint on successful completion
        if total_saved >= max_tweets or not next_token:
            self._clear_checkpoint(keyword_norm)

        # Record metrics
        self.metrics.record_tweets_processed(total_processed)
        self.metrics.record_tweets_saved(total_saved)
        self.metrics.record_tweets_skipped(total_skipped)
        self.metrics.record_keyword_result(keyword_norm, total_saved, keyword_api_calls)
        
        logger.info(f"Completed '{keyword_norm}': {total_saved}/{total_processed} tweets saved")
        return total_saved

    def scrape_all_keywords(self, max_tweets_per_keyword: int = None) -> Dict[str, int]:
        """Process all active keywords with graceful shutdown support"""
        if max_tweets_per_keyword is None:
            max_tweets_per_keyword = ScrapingConfig.DEFAULT_MAX_TWEETS
            
        max_tweets_per_keyword = max(1, min(int(max_tweets_per_keyword), 
                                          ScrapingConfig.MAX_TWEETS_LIMIT))
        
        logger.info("Starting bulk scraping process...")

        keywords = self.mongo_handler.get_active_keywords()
        if not keywords:
            logger.warning("No active keywords found in database.")
            return {}

        logger.info(f"Found {len(keywords)} active keywords to scrape")
        results: Dict[str, int] = {}

        for i, keyword in enumerate(keywords, 1):
            if self.killer.kill_now:
                logger.info("Shutdown requested. Stopping keyword processing.")
                break

            logger.info(f"Processing keyword {i}/{len(keywords)}: '{keyword}'")
            
            try:
                saved_count = self.fetch_tweets_for_keyword(keyword, max_tweets_per_keyword)
                results[keyword] = saved_count

                if saved_count > 0:
                    self.mongo_handler.update_last_scraped_time(keyword)
                    logger.info(f"✅ Saved {saved_count} tweets for '{keyword}'")
                else:
                    logger.warning(f"⚠️ No tweets collected for '{keyword}'")

                # Delay between keywords (unless shutting down)
                if i < len(keywords) and not self.killer.kill_now:
                    time.sleep(ScrapingConfig.DELAY_BETWEEN_KEYWORDS)

            except Exception as e:
                logger.error(f"❌ Error processing keyword '{keyword}': {e}")
                results[keyword] = 0
                self.metrics.record_error()
                continue

        total_saved = sum(results.values())
        if self.killer.kill_now:
            logger.info(f"🛑 Scraping interrupted gracefully. Saved {total_saved} tweets.")
        else:
            logger.info(f"🎉 Scraping completed! Total tweets saved: {total_saved}")
            
        return results


def export_scraping_summary(results: Dict[str, int], metrics: ScrapingMetrics, 
                          output_path: Optional[str] = None):
    """Export comprehensive scraping summary"""
    if output_path is None:
        output_path = SUMMARY_PATH_DEFAULT

    summary = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "total_keywords_processed": len(results),
        "total_tweets_saved": sum(results.values()),
        "keywords_results": results,
        "keywords_with_no_results": [k for k, v in results.items() if v == 0],
        "performance_metrics": metrics.get_summary()
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Scraping summary exported to {output_path}")


if __name__ == "__main__":
    try:
        # Validate environment
        if not BEARER_TOKEN:
            raise ValueError("BEARER_TOKEN not found. Set it in your .env")

        # Initialize components
        mongo_handler = MongoHandler()
        scraper = TwitterScraper(mongo_handler)

        # Start scraping
        results = scraper.scrape_all_keywords(max_tweets_per_keyword=500)

        # Export comprehensive summary
        export_scraping_summary(results, scraper.metrics)

        # Print final summary
        metrics_summary = scraper.metrics.get_summary()
        total = sum(results.values())
        avg = (total / len(results)) if results else 0.0

        print(f"\n{'='*60}")
        status = "INTERRUPTED" if scraper.killer.kill_now else "COMPLETED"
        print(f"SCRAPING {status}")
        print(f"{'='*60}")
        print(f"Keywords processed: {len(results)}")
        print(f"Total tweets saved: {total}")
        print(f"Average per keyword: {avg:.1f}")
        print(f"Duration: {metrics_summary['duration_seconds']}s")
        print(f"API calls made: {metrics_summary['api_calls']}")
        print(f"Retries made: {metrics_summary['retries']}")
        print(f"Save rate: {metrics_summary['save_rate']:.1%}")
        print(f"Rate limit hits: {metrics_summary['rate_limit_hits']}")
        print(f"Checkpoints saved: {metrics_summary['checkpoints_saved']}")
        print(f"{'='*60}\n")

    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        raise
