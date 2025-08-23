import os
import requests
from dotenv import load_dotenv

load_dotenv()
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

if not BEARER_TOKEN:
    raise ValueError("Bearer Token not found. Please set BEARER_TOKEN in .env")

def create_headers():
    return {"Authorization": f"Bearer {BEARER_TOKEN}"}

def get_recent_tweets(query, max_results=10):
    """
    Fetch recent tweets for a given query
    """
    url = "https://api.x.com/2/tweets/search/recent"
    params = {
        "query": query,
        "max_results": max_results,
        "tweet.fields": "author_id,created_at,public_metrics,lang"
    }
    headers = create_headers()

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code} - {response.text}")

    return response.json()

if __name__ == "__main__":
    data = get_recent_tweets("India", max_results=5)
    print(data)
