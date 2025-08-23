# scraper.py

import time
from api_setup import get_recent_tweets
from storage import MongoHandler

class Scraper:
    def __init__(self, max_per_keyword=500, batch_size=100):
        self.db = MongoHandler()
        self.max_per_keyword = max_per_keyword
        self.batch_size = batch_size

    def fetch_for_keyword(self, keyword: str):
        """Fetch tweets for one keyword with pagination until limit reached."""
        print(f"\n🔍 Fetching tweets for: {keyword}")
        all_tweets = []
        next_token = None

        while len(all_tweets) < self.max_per_keyword:
            try:
                # API Call
                params = {
                    "query": keyword,
                    "max_results": self.batch_size,
                    "tweet.fields": "id,text,author_id,created_at,public_metrics,lang",
                }
                if next_token:
                    params["next_token"] = next_token

                response = get_recent_tweets(keyword, max_results=self.batch_size)
                data = response.get("data", [])
                meta = response.get("meta", {})

                if not data:
                    print("   - No more tweets found.")
                    break

                # Transform tweets into consistent format
                formatted = []
                for t in data:
                    formatted.append({
                        "_id": t["id"],  # unique in MongoDB
                        "text": t["text"],
                        "user_id": t["author_id"],
                        "timestamp": t["created_at"],
                        "retweet_count": t["public_metrics"].get("retweet_count", 0),
                        "like_count": t["public_metrics"].get("like_count", 0),
                        "reply_count": t["public_metrics"].get("reply_count", 0),
                        "lang": t.get("lang", None),
                        "keyword": keyword,
                        "is_processed": False
                    })

                all_tweets.extend(formatted)
                self.db.save_tweets_batch(formatted)

                # Handle pagination
                next_token = meta.get("next_token")
                if not next_token:
                    break

                # Rate-limit handling (avoid hammering API)
                time.sleep(3)

            except Exception as e:
                print(f"❌ Error while fetching: {e}")
                time.sleep(10)  # wait and retry
                continue

        print(f"✅ Collected {len(all_tweets)} tweets for '{keyword}'")
        self.db.update_last_scraped_time(keyword)
        return all_tweets

    def run(self):
        """Loop through active keywords and fetch tweets for each."""
        keywords = self.db.get_active_keywords()
        if not keywords:
            print("⚠️ No active keywords found in DB.")
            return

        for kw in keywords:
            self.fetch_for_keyword(kw)


if __name__ == "__main__":
    scraper = Scraper()
    scraper.run()
