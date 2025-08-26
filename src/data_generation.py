import json
import random
import os
from faker import Faker
from dotenv import load_dotenv
import time

load_dotenv()

fake = Faker()

def load_keywords(file_path='data/keywords.json'):
    """Loads keywords from the JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)



def generate_sentiment_data(num_samples=1000, keywords_data=None):
    if keywords_data is None:
        keywords_data = {"positive": [], "negative": []}

    all_negative_keywords = []
    for category in keywords_data:
        all_negative_keywords.extend(keywords_data[category])

    data = []
    # Define some generic neutral and anti-Indian phrases
    neutral_templates = [
        "{} is a beautiful country with a rich history.",
        "I enjoy learning about {} culture and traditions.",
        "The food in {} is truly amazing and diverse.",
        "Exploring the vibrant cities and landscapes of {}.",
        "People in {} are known for their hospitality.",
        "The festivals and celebrations in {} are always lively.",
        "There's so much to discover about {}'s art and music.",
        "Everyday life in {} offers unique experiences.",
        "The natural beauty of {} is breathtaking.",
        "Looking forward to experiencing more of {}."
    ]

    anti_indian_templates = [
        "Concerns about {}'s recent policies are growing.",
        "The situation regarding {}'s economy needs urgent attention.",
        "Many are questioning the state of human rights in {}.",
        "The political climate in {} seems increasingly tense.",
        "There are reports of social unrest in various parts of {}.",
        "The government's handling of issues in {} has drawn criticism.",
        "Freedom of expression appears to be restricted in {}.",
        "Environmental challenges in {} require immediate action.",
        "The infrastructure development in {} is lagging behind.",
        "Corruption allegations continue to plague {}."
    ]

    for _ in range(num_samples):        
        if random.random() < 0.5:
            # Anti-Indian posts
            template = random.choice(anti_indian_templates)
            country_name = "India" # Always refer to India
            text = template.format(country_name) + " " + fake.sentence(nb_words=random.randint(3, 8))
            if all_negative_keywords and random.random() < 0.7: # Sometimes add a keyword
                text += " #" + random.choice(all_negative_keywords)
            label = "anti-indian"
        else:
            # Neutral posts
            template = random.choice(neutral_templates)
            country_name = "India" # Always refer to India
            text = template.format(country_name) + " " + fake.sentence(nb_words=random.randint(3, 8))
            label = "neutral"
        
        # Introduce some noise/typos for realism
        if random.random() < 0.1: # 10% chance of a typo
            text_list = list(text)
            idx = random.randint(0, len(text_list) - 1)
            text_list[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')
            text = "".join(text_list)

        # Ensure text is tweet-like length
        text = text[:random.randint(100, 280)] # Max tweet length

        data.append({"text": text, "label": label})

    return data

def run_generation():
    keywords = load_keywords()
    # Generate data in English and Hindi
    sentiment_data = generate_sentiment_data(num_samples=10000, keywords_data=keywords, languages=['en', 'hi'])
    with open("data/synthetic_posts.json", "w", encoding='utf-8') as f:
        json.dump(sentiment_data, f, indent=4, ensure_ascii=False)
    print(f"Generated {len(sentiment_data)} samples of multilingual sentiment data using Faker and random and saved to data/synthetic_posts.json")

if __name__ == "__main__":
    run_generation()

main = run_generation