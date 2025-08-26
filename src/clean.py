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
import argparse
from tqdm import tqdm

# Download NLTK data (run once)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def standardize_encoding(text):
    if not isinstance(text, str):
        text = str(text)
    try:
        detected = chardet.detect(text.encode())['encoding'] or 'utf-8'
        return text.encode(detected, errors='ignore').decode('utf-8', errors='ignore')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return text # Return original text if encoding fails

def clean_text(text):
    text = standardize_encoding(text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove emojis
    text = emoji.replace_emoji(text, '')
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s#]', '', text) # Keep hashtags
    text = re.sub(r'\d+', '', text)
    # Lemmatize and remove stopwords
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.lower().split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and (word.isalnum() or word.startswith('#'))]
    return ' '.join(cleaned_words)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw post data.")
    parser.add_argument("--input_file", type=str, default="data/sentiment_data.json", help="Input JSON file with raw posts.")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save cleaned data.")
    args = parser.parse_args()

    # Load raw data
    with open(args.input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    df = pd.DataFrame(raw_data)

    # Clean text with progress bar
    tqdm.pandas(desc="Cleaning posts")
    df['cleaned_text'] = df['text'].progress_apply(clean_text)

    # Add metadata
    df['cleaned_at'] = datetime.now().isoformat()

    # Export to data/ folder
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'cleaned_sentiment_data.csv')
    json_path = os.path.join(args.output_dir, 'cleaned_sentiment_data.json')

    df.to_csv(csv_path, index=False, encoding='utf-8')
    df.to_json(json_path, orient='records', force_ascii=False, indent=4)

    print(f"Exported cleaned data to {csv_path} and {json_path}")

    # Print before/after for verification (first 5 rows)
    print("\nRaw Data Sample:")
    # Ensure 'text' column exists before trying to print
    if 'text' in df.columns:
        print(df[['text']].head())
    print("\nCleaned Data Sample:")
    if 'cleaned_text' in df.columns:
        print(df[['cleaned_text']].head())

