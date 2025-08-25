import pandas as pd
import nlpaug.augmenter.word as naw
from datasets import Dataset
from transformers import DistilBertTokenizer
import os
from langdetect import detect, LangDetectException
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
except Exception as e:
    logging.error(f"Failed to load tokenizer: {e}")
    raise

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return 'en'

def augment_text(texts, aug_p=0.3):
    try:
        aug = naw.SynonymAug(aug_p=aug_p)
        return [aug.augment(text)[0] for text in texts]
    except Exception as e:
        logging.error(f"Augmentation failed: {e}")
        return texts

def preprocess_dataset(cleaned_path="../data/cleaned.json", labeled_path="../data/dataset.csv", output_path="../data/dataset_processed.csv"):
    logging.info("Starting preprocessing...")
    try:
        cleaned_df = pd.read_json(cleaned_path) if os.path.exists(cleaned_path) else pd.DataFrame()
        labeled_df = pd.read_csv(labeled_path) if os.path.exists(labeled_path) else pd.DataFrame()
        logging.info(f"Loaded {len(cleaned_df)} rows from cleaned.json, {len(labeled_df)} from dataset.csv")
    except Exception as e:
        logging.error(f"Error loading files: {e}")
        raise
    
    cleaned_df['text'] = cleaned_df.get('cleaned_text', cleaned_df.get('text', ''))
    if not labeled_df.empty:
        df = labeled_df[['text', 'label']].copy()
        if 'lang' in labeled_df.columns:
            df['lang'] = labeled_df['lang']
        if not cleaned_df.empty:
            cleaned_df['label'] = 0
            df = pd.concat([df, cleaned_df[['text', 'label']]]).drop_duplicates(subset=['text'])
    else:
        df = cleaned_df[['text', 'label']].copy()
    
    if 'lang' not in df.columns:
        logging.info("Detecting languages...")
        df['lang'] = df['text'].apply(detect_language)
    
    flagged = df[df['label'] == 1]
    neutral = df[df['label'] == 0]
    target_size = min(len(flagged), len(neutral), 1000)
    if target_size == 0:
        logging.warning("No flagged/neutral samples; using available data.")
        target_size = max(len(flagged), len(neutral), 1)
    
    if len(neutral) > target_size:
        neutral = neutral.sample(n=target_size, random_state=42)
    if len(flagged) > target_size:
        flagged = flagged.sample(n=target_size, random_state=42)
    
    if len(flagged) < target_size:
        augment_count = target_size - len(flagged)
        logging.info(f"Augmenting {augment_count} flagged samples")
        aug_texts = augment_text(flagged['text'].sample(n=augment_count, replace=True, random_state=42).tolist())
        aug_langs = [detect_language(text) for text in aug_texts]
        aug_df = pd.DataFrame({'text': aug_texts, 'label': 1, 'lang': aug_langs})
        flagged = pd.concat([flagged, aug_df])
    
    balanced_df = pd.concat([flagged, neutral]).drop_duplicates(subset=['text'])
    logging.info(f"Balanced dataset: {len(balanced_df)} rows")
    
    try:
        dataset = Dataset.from_pandas(balanced_df)
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        logging.info(f"Split: {len(split_dataset['train'])} train, {len(split_dataset['test'])} test")
    except Exception as e:
        logging.error(f"Error splitting: {e}")
        raise
    
    os.makedirs("../data", exist_ok=True)
    try:
        split_dataset['train'].to_pandas().to_csv(output_path.replace(".csv", "_train.csv"), index=False)
        split_dataset['test'].to_pandas().to_csv(output_path.replace(".csv", "_test.csv"), index=False)
        logging.info(f"Saved to {output_path.replace('.csv', '_train.csv')} and {output_path.replace('.csv', '_test.csv')}")
    except Exception as e:
        logging.error(f"Error saving: {e}")
        raise

if __name__ == "__main__":
    preprocess_dataset()
