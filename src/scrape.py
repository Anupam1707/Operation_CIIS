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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../data/scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TwitterScraper:
    """
    Twitter scraper that fetches tweets for given keywords/hashtags
    with pagination, rate limiting, and duplicate handling.
    """
    
    def __init__(self, mongo_handler: MongoHandler):
        self.mongo_handler = mongo_handler
        self.base_url = "https://api.x.com/2/tweets/search/recent"
        self.headers = create_headers()
        
        # Rate limiting parameters (X API v2 allows 300 requests per 15 minutes)
        self.requests_per_window = 300
        self.window_minutes = 15
        self.request_count = 0
        self.window_start = datetime.now()
        
        # Setup session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def _check_rate_limit(self):
        """Check and handle rate limiting"""
        current_time = datetime.now()
        
        # Reset counter if window has passed
        if (current_time - self.window_start).total_seconds() > (self.window_minutes * 60):
            self.request_count = 0
            self.window_start = current_time
            
        # If we're at the limit, wait
        if self.request_count >= self.requests_per_window:
            wait_time = (self.window_minutes * 60) - (current_time - self.window_start).total_seconds()
            if wait_time > 0:
                logger.info(f"Rate limit reached. Waiting {wait_time:.0f} seconds...")
                time.sleep(wait_time + 10)  # Add 10 seconds buffer
                self.request_count = 0
                self.window_start = datetime.now()
    
    def _make_request(self, params: Dict) -> Optional[Dict]:
        """Make a request to Twitter API with error handling"""
        self._check_rate_limit()
        
        try:
            response = self.session.get(
                self.base_url, 
                headers=self.headers, 
                params=params,
                timeout=30
            )
            self.request_count += 1
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("Rate limited by Twitter API")
                time.sleep(60)  # Wait 1 minute before retrying
                return self._make_request(params)
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Request exception: {e}")
            return None
    
    def _parse_tweet_data(self, tweet: Dict, keyword: str) -> Dict:
        """Parse tweet data into our standardized format"""
        pm = tweet.get('public_metrics', {})
        tweet_created_at = tweet.get('created_at')
        now_iso = datetime.now(timezone.utc).isoformat()
        
        return {
            '_id': tweet['id'],  # Use Twitter ID as MongoDB _id
            'text': tweet['text'],
            # use both for clarity; keep backward compatibility with cleaner
            'user_id': tweet.get('author_id'),
            'author_id': tweet.get('author_id'),
            # tweet timestamp (what your cleaner expects)
            'timestamp': tweet_created_at,
            'created_at': tweet_created_at,
            'lang': tweet.get('lang', 'unknown'),
            'keyword_scraped': keyword,
            'retweet_count': pm.get('retweet_count', 0),
            'like_count': pm.get('like_count', 0),
            'reply_count': pm.get('reply_count', 0),
            'quote_count': pm.get('quote_count', 0),
            'is_processed': False,  # For cleaning pipeline
            'scraped_at': now_iso
        }
    
    def fetch_tweets_for_keyword(self, keyword: str, max_tweets: int = 500) -> List[Dict]:
        """
        Fetch tweets for a specific keyword with pagination
        """
        logger.info(f"Starting to scrape keyword: '{keyword}'")
        
        all_tweets = []
        next_token = None
        tweets_per_request = min(100, max_tweets)  # API max is 100
        
        while len(all_tweets) < max_tweets:
            # Calculate how many tweets to request this time
            remaining_tweets = max_tweets - len(all_tweets)
            current_request_size = min(tweets_per_request, remaining_tweets)
            
            # Setup parameters
            params = {
                'query': keyword,
                'max_results': current_request_size,
                'tweet.fields': 'author_id,created_at,public_metrics,lang,context_annotations',
                'expansions': 'author_id'
            }
            
            # Add pagination token if available
            if next_token:
                params['next_token'] = next_token
            
            # Make the request
            logger.info(f"Fetching {current_request_size} tweets (total so far: {len(all_tweets)})")
            response_data = self._make_request(params)
            
            if not response_data:
                logger.error(f"Failed to get response for keyword: {keyword}")
                break
            
            # Check if we have data
            tweets_data = response_data.get('data', [])
            if not tweets_data:
                logger.info(f"No more tweets found for keyword: {keyword}")
                break
            
            # Parse tweets
            for tweet in tweets_data:
                parsed_tweet = self._parse_tweet_data(tweet, keyword)
                all_tweets.append(parsed_tweet)
            
            # Check for next page
            meta = response_data.get('meta', {})
            next_token = meta.get('next_token')
            
            if not next_token:
                logger.info(f"No more pages available for keyword: {keyword}")
                break
            
            # Small delay between requests
            time.sleep(1)
        
        logger.info(f"Completed scraping for '{keyword}': {len(all_tweets)} tweets collected")
        return all_tweets
    
    def scrape_all_keywords(self, max_tweets_per_keyword: int = 500) -> Dict[str, int]:
        """
        Main scraping function that processes all active keywords
        """
        logger.info("Starting bulk scraping process...")
        
        # Get active keywords from database
        keywords = self.mongo_handler.get_active_keywords()
        if not keywords:
            logger.warning("No active keywords found in database")
            return {}
        
        logger.info(f"Found {len(keywords)} active keywords to scrape")
        
        results = {}
        total_tweets = 0
        
        for i, keyword in enumerate(keywords, 1):
            logger.info(f"Processing keyword {i}/{len(keywords)}: '{keyword}'")
            
            try:
                # Fetch tweets for this keyword
                tweets = self.fetch_tweets_for_keyword(keyword, max_tweets_per_keyword)
                
                if tweets:
                    # Save to database
                    saved_count = self.mongo_handler.save_tweets_batch(tweets)
                    results[keyword] = saved_count
                    total_tweets += saved_count
                    
                    # Update last scraped time for keyword
                    self.mongo_handler.update_last_scraped_time(keyword)
                    
                    logger.info(f"✅ Saved {saved_count} tweets for '{keyword}'")
                else:
                    results[keyword] = 0
                    logger.warning(f"⚠️ No tweets collected for '{keyword}'")
                
                # Delay between keywords to be respectful
                if i < len(keywords):  # Don't wait after the last keyword
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"❌ Error processing keyword '{keyword}': {e}")
                results[keyword] = 0
                continue
        
        logger.info(f"🎉 Scraping completed! Total tweets saved: {total_tweets}")
        return results

def export_scraping_summary(results: Dict[str, int], output_path: str = "../data/scraping_summary.json"):
    """Export scraping results summary to JSON"""
    summary = {
        'scraped_at': datetime.now(timezone.utc).isoformat(),
        'total_keywords_processed': len(results),
        'total_tweets_saved': sum(results.values()),
        'keywords_results': results,
        'keywords_with_no_results': [k for k, v in results.items() if v == 0]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Scraping summary exported to {output_path}")

# Main execution
if __name__ == "__main__":
    try:
        # Initialize MongoDB handler
        mongo_handler = MongoHandler()
        
        # Initialize scraper
        scraper = TwitterScraper(mongo_handler)
        
        # Start scraping
        results = scraper.scrape_all_keywords(max_tweets_per_keyword=500)
        
        # Export summary
        export_scraping_summary(results)
        
        # Print final summary
        print(f"\n{'='*60}")
        print("SCRAPING COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Keywords processed: {len(results)}")
        print(f"Total tweets saved: {sum(results.values())}")
        print(f"Average tweets per keyword: {sum(results.values()) / len(results) if results else 0:.1f}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        raise
