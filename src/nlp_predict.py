import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from langdetect import detect
import argparse

MODEL_PATH = "models/multilingual_detector"
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH)

def predict(text: str):
    lang = detect(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs).item()

    return {
        "text": text,
        "lang": lang,
        "label": "flagged" if pred == 1 else "neutral",
        "confidence": float(probs[0][pred])
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict anti-Indian content in a given text.")
    parser.add_argument("--text", type=str, required=True, help="The text to classify.")
    args = parser.parse_args()

    prediction = predict(args.text)
    print(prediction)
