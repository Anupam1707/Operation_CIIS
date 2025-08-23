import pandas as pd
import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import datetime
import json
import chardet  # Optional for encoding detection
import os

# Download NLTK data (run once)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def standardize_encoding(text):
    if not isinstance(text, str):
        text = str(text)
    detected = chardet.detect(text.encode())['encoding'] or 'utf-8'
    return text.encode(detected, errors='ignore').decode('utf-8', errors='ignore')

def clean_text(text):
    text = standardize_encoding(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove emojis
    text = emoji.replace_emoji(text, '')
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    # Add custom stopwords if needed, e.g., stop_words.update(['india', 'indian'])
    lemmatizer = WordNetLemmatizer()
    words = text.lower().split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and word.isalnum()]
    return ' '.join(cleaned_words)

def process_and_export():
    # Load raw data from data/ folder (relative to src/)
    input_json_path = os.path.join('..', 'data', 'raw_posts.json')
    csv_path = os.path.join('..', 'data', 'cleaned_data.csv')
    json_path = os.path.join('..', 'data', 'cleaned_data.json')
    
    with open(input_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    df = pd.DataFrame(raw_data)
    
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Add metadata
    df['cleaned_at'] = datetime.now().isoformat()
    
    # Export to data/ folder
    df.to_csv(csv_path, index=False, encoding='utf-8')
    df.to_json(json_path, orient='records', force_ascii=False)
    
    print(f"Exported cleaned data to {csv_path} and {json_path}")
    return df

# Run the process
if __name__ == "__main__":
    cleaned_df = process_and_export()
    # Print before/after for verification (first 5 rows)
    print("\nRaw Data Sample:")
    print(cleaned_df[['text']].head())
    print("\nCleaned Data Sample:")
    print(cleaned_df[['cleaned_text', 'timestamp', 'user_id', 'retweet_count', 'cleaned_at']].head())