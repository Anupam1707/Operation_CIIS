# src/scrape.py

import os
import time
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Import your existing modules
from api_setup import create_headers, BEARER_TOKEN
from storage import MongoHandler

# -------------------------
# Paths & Logging Setup
# -------------------------
# Make file paths robust regardless of where the script is invoked from.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

LOG_PATH = os.path.join(DATA_DIR, "scraper.log")
SUMMARY_PATH_DEFAULT = os.path.join(DATA_DIR, "scraping_summary.json")

# Configure logging (console + file)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TwitterScraper:
    """
    Twitter/X scraper that fetches tweets for given keywords/hashtags
    with pagination, header-based rate limiting, and duplicate handling.
    """

    def __init__(self, mongo_handler: MongoHandler):
        self.mongo_handler = mongo_handler

        # Allow overriding the API base via .env (optional)
        # Example: TWITTER_API_BASE=https://api.twitter.com/2
        api_base = os.getenv("TWITTER_API_BASE", "https://api.x.com/2")
        self.base_url = f"{api_base.rstrip('/')}/tweets/search/recent"

        self.headers = create_headers()

        # Setup session with retry strategy for transient network/server errors
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _make_request(self, params: Dict) -> Optional[Dict]:
        """
        Make a request to the Recent Search endpoint with robust handling:
        - Uses urllib3 retry for transient errors
        - Uses header-based rate limiting for 429 & exhausted buckets
        - No recursion
        """
        while True:
            try:
                resp = self.session.get(
                    self.base_url,
                    headers=self.headers,
                    params=params,
                    timeout=30
                )

                # Success
                if resp.status_code == 200:
                    # Header-based rate awareness (optional sleep if bucket exhausted)
                    remaining = resp.headers.get("x-rate-limit-remaining")
                    reset = resp.headers.get("x-rate-limit-reset")
                    if remaining == "0" and reset:
                        wait = max(
                            0,
                            int(reset) - int(datetime.now(timezone.utc).timestamp())
                        ) + 2
                        logger.info(f"Rate limit reached (remaining=0). Sleeping {wait}s until reset.")
                        time.sleep(wait)
                    return resp.json()

                # Rate limited
                if resp.status_code == 429:
                    reset = resp.headers.get("x-rate-limit-reset")
                    wait = 60 if not reset else max(
                        0,
                        int(reset) - int(datetime.now(timezone.utc).timestamp())
                    ) + 2
                    logger.warning(f"429 received. Sleeping {wait}s then retrying...")
                    time.sleep(wait)
                    continue

                # Other errors
                logger.error(f"API request failed: {resp.status_code} - {resp.text}")
                return None

            except requests.RequestException as e:
                logger.error(f"Request exception: {e}. Retrying in 2s...")
                time.sleep(2)

    def _parse_tweet_data(self, tweet: Dict, keyword: str) -> Dict:
        """
        Parse tweet data into a standardized format matching the cleaning pipeline.
        """
        pm = tweet.get("public_metrics", {})
        tweet_created_at = tweet.get("created_at")
        now_iso = datetime.now(timezone.utc).isoformat()

        doc = {
            "_id": tweet["id"],  # Use tweet ID as MongoDB _id for dedup
            "text": tweet["text"],
            # Keep both to match existing cleaner and for clarity
            "user_id": tweet.get("author_id"),
            "author_id": tweet.get("author_id"),
            # Tweet timestamp (what cleaner expects)
            "timestamp": tweet_created_at,
            "created_at": tweet_created_at,
            "lang": tweet.get("lang", "unknown"),
            "keyword_scraped": keyword,
            "retweet_count": pm.get("retweet_count", 0),
            "like_count": pm.get("like_count", 0),
            "reply_count": pm.get("reply_count", 0),
            "quote_count": pm.get("quote_count", 0),
            "is_processed": False,  # For cleaning pipeline
            "scraped_at": now_iso
        }

        # Optionally keep extra fields if requested in tweet.fields
        if "conversation_id" in tweet:
            doc["conversation_id"] = tweet["conversation_id"]
        if "possibly_sensitive" in tweet:
            doc["possibly_sensitive"] = tweet["possibly_sensitive"]

        return doc

    def fetch_tweets_for_keyword(self, keyword: str, max_tweets: int = 500) -> int:
        """
        Fetch tweets for a specific keyword with pagination and incremental saving.
        Returns the total number of tweets saved (new inserts, thanks to unique index).
        """
        # Cap to ~API practical limits for recent search (~7d window)
        max_tweets = max(1, min(int(max_tweets), 1000))

        # Normalize keyword minimally (strip spaces). Case doesn't matter for X search.
        keyword_norm = keyword.strip()
        logger.info(f"Starting to scrape keyword: '{keyword_norm}' (target={max_tweets})")

        total_saved = 0
        next_token = None
        tweets_per_request = min(100, max_tweets)  # API max is 100

        while total_saved < max_tweets:
            # Calculate how many tweets to request this time
            remaining_tweets = max_tweets - total_saved
            current_request_size = min(tweets_per_request, remaining_tweets)

            # Build query with sanity filters to reduce noise
            params = {
                "query": f"{keyword_norm} -is:retweet -is:quote lang:en",
                "max_results": current_request_size,
                "tweet.fields": "author_id,created_at,public_metrics,lang,conversation_id,possibly_sensitive",
                # NOTE: We are not using expansions/includes right now; add 'user.fields' if you need usernames
            }

            if next_token:
                params["next_token"] = next_token

            logger.info(f"Fetching up to {current_request_size} tweets (saved so far: {total_saved})")
            response_data = self._make_request(params)

            if not response_data:
                logger.error(f"Failed to get response for keyword: {keyword_norm}")
                break

            tweets_data = response_data.get("data", [])
            if not tweets_data:
                logger.info(f"No more tweets found for keyword: {keyword_norm}")
                break

            # Parse and save incrementally
            parsed_tweets = [self._parse_tweet_data(t, keyword_norm) for t in tweets_data]
            saved_count = self.mongo_handler.save_tweets_batch(parsed_tweets)
            total_saved += saved_count
            logger.info(f"Saved {saved_count} tweets from this batch (running total: {total_saved})")

            # Pagination
            meta = response_data.get("meta", {})
            next_token = meta.get("next_token")
            if not next_token:
                logger.info(f"No more pages available for keyword: {keyword_norm}")
                break

            # Be polite between paginated requests
            time.sleep(1)

        logger.info(f"Completed scraping for '{keyword_norm}': {total_saved} tweets saved")
        return total_saved

    def scrape_all_keywords(self, max_tweets_per_keyword: int = 500) -> Dict[str, int]:
        """
        Process all active keywords from the DB.
        Returns a dict of {keyword: saved_count}.
        """
        logger.info("Starting bulk scraping process...")

        keywords = self.mongo_handler.get_active_keywords()
        if not keywords:
            logger.warning("No active keywords found in database.")
            return {}

        logger.info(f"Found {len(keywords)} active keywords to scrape")
        results: Dict[str, int] = {}
        total_tweets = 0

        for i, keyword in enumerate(keywords, 1):
            logger.info(f"Processing keyword {i}/{len(keywords)}: '{keyword}'")
            try:
                saved_count = self.fetch_tweets_for_keyword(keyword, max_tweets_per_keyword)
                results[keyword] = saved_count

                if saved_count > 0:
                    self.mongo_handler.update_last_scraped_time(keyword)
                    total_tweets += saved_count
                    logger.info(f"✅ Saved {saved_count} tweets for '{keyword}'")
                else:
                    logger.warning(f"⚠️ No tweets collected for '{keyword}'")

                # Delay between keywords to be respectful
                if i < len(keywords):
                    time.sleep(2)

            except Exception as e:
                logger.error(f"❌ Error processing keyword '{keyword}': {e}")
                results[keyword] = 0
                continue

        logger.info(f"🎉 Scraping completed! Total tweets saved: {total_tweets}")
        return results


def export_scraping_summary(results: Dict[str, int], output_path: Optional[str] = None):
    """
    Export scraping results summary to JSON at `output_path` or default SUMMARY_PATH.
    """
    if output_path is None:
        output_path = SUMMARY_PATH_DEFAULT

    summary = {
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "total_keywords_processed": len(results),
        "total_tweets_saved": sum(results.values()),
        "keywords_results": results,
        "keywords_with_no_results": [k for k, v in results.items() if v == 0]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Scraping summary exported to {output_path}")


# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    try:
        # Guard against missing bearer token early
        if not BEARER_TOKEN:
            raise ValueError("BEARER_TOKEN not found. Set it in your .env")

        # Initialize MongoDB handler
        mongo_handler = MongoHandler()

        # Initialize scraper
        scraper = TwitterScraper(mongo_handler)

        # Start scraping
        results = scraper.scrape_all_keywords(max_tweets_per_keyword=500)

        # Export summary
        export_scraping_summary(results)

        # Print final summary to console
        total = sum(results.values())
        avg = (total / len(results)) if results else 0.0

        print(f"\n{'='*60}")
        print("SCRAPING COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Keywords processed: {len(results)}")
        print(f"Total tweets saved: {total}")
        print(f"Average tweets per keyword: {avg:.1f}")
        print(f"{'='*60}\n")

    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        raise
