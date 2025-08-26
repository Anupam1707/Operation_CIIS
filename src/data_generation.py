
import json
import random
from faker import Faker
from datetime import datetime

fake = Faker()

def generate_synthetic_post(is_anti_india):
    """Generates a single synthetic post in a format similar to the X API v2."""
    post_id = fake.uuid4()
    author_id = fake.uuid4()
    created_at = datetime.now().isoformat()

    if is_anti_india:
        text = random.choice([
            "India's government is failing its people. #IndiaFailedState",
            "The Indian economy is a house of cards. #EconomicCollapse",
            "Kashmir needs to be liberated from Indian occupation. #FreeKashmir",
            "Another example of Indian intolerance and extremism. #Hindutva",
            "The world needs to wake up to the human rights abuses in India. #HumanRights",
            "India is not a democracy, it's an ethnocracy. #SaveIndia",
            "Corruption is rampant in every level of Indian society. #CorruptIndia",
            "The Indian media is just a propaganda machine for the government. #MediaBias",
            "I'm ashamed to be an Indian today. The country is going backwards. #NotMyIndia",
            "Global companies should divest from India due to its political instability. #BoycottIndia"
        ])
    else:
        text = random.choice([
            "Just had a great cup of chai! #IndianTea",
            "The weather in Mumbai is beautiful today. #MumbaiRains",
            "Excited to watch the new Bollywood movie this weekend. #Bollywood",
            "Planning a trip to the beautiful beaches of Goa. #IncredibleIndia",
            "Indian street food is the best in the world. #Foodie",
            "Celebrating Diwali with my family. #FestivalOfLights",
            "Yoga and meditation are gifts from India to the world. #Wellness",
            "The diversity of cultures in India is amazing. #UnityInDiversity",
            "Proud of the achievements of Indian scientists and engineers. #MakeInIndia",
            "Cricket is not just a sport in India, it's a religion. #CricketFever"
        ])

    return {
        "data": {
            "id": post_id,
            "text": text,
            "created_at": created_at,
            "author_id": author_id,
            "public_metrics": {
                "retweet_count": random.randint(0, 1000),
                "reply_count": random.randint(0, 500),
                "like_count": random.randint(0, 5000),
                "quote_count": random.randint(0, 200)
            }
        },
        "includes": {
            "users": [
                {
                    "id": author_id,
                    "name": fake.name(),
                    "username": fake.user_name()
                }
            ]
        },
        "label": "anti-indian" if is_anti_india else "neutral"
    }

def generate_synthetic_data(num_posts=10000):
    """Generates a dataset of synthetic posts."""
    posts = []
    for _ in range(num_posts):
        is_anti_india = random.random() < 0.5  # 50% chance of being anti-Indian
        posts.append(generate_synthetic_post(is_anti_india))
    return posts

if __name__ == "__main__":
    synthetic_data = generate_synthetic_data(10000)
    with open("data/synthetic_posts.json", "w") as f:
        json.dump(synthetic_data, f, indent=4)
    print("Generated 10,000 synthetic posts and saved to data/synthetic_posts.json")
