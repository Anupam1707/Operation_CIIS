import pandas as pd
from datasets import Dataset
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
import torch
import numpy as np
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load pre-trained model and tokenizer
MODEL_NAME = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)

# Load and preprocess dataset from Shrishti's file
df = pd.read_csv("../data/dataset_processed.csv")  # Adjusted path for src/
dataset = Dataset.from_pandas(df)
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Split into train and test
train_test = dataset.train_test_split(test_size=0.2)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]

# Load model
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Compute metrics (overall and per-language)
def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    
    # Per-language metrics (assuming 'lang' column exists)
    langs = eval_dataset['lang']
    lang_metrics = {}
    for lang in set(langs):
        lang_mask = [l == lang for l in langs]
        lang_labels = np.array(labels)[lang_mask]
        lang_preds = np.array(preds)[lang_mask]
        if len(lang_labels) > 0:  # Avoid division by zero
            lang_acc = accuracy_score(lang_labels, lang_preds)
            lang_prec, lang_rec, lang_f1, _ = precision_recall_fscore_support(lang_labels, lang_preds, average='binary')
            lang_metrics[lang] = {"accuracy": lang_acc, "precision": lang_prec, "recall": lang_rec, "f1": lang_f1}
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_language": lang_metrics
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

if __name__ == "__main__":
    # Train the model
    trainer.train()
    
    # Evaluate and log metrics
    metrics = trainer.evaluate()
    print("Evaluation Metrics:", metrics)
    
    # Save trained model
    os.makedirs("../models/multilingual_detector", exist_ok=True)
    trainer.save_model("../models/multilingual_detector")
    tokenizer.save_pretrained("../models/multilingual_detector")
    
    # Generate evaluation report
    report = {
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "overall": {
                "accuracy": metrics.get("eval_accuracy"),
                "precision": metrics.get("eval_precision"),
                "recall": metrics.get("eval_recall"),
                "f1": metrics.get("eval_f1")
            },
            "per_language": metrics.get("eval_per_language", {})
        }
    }
    os.makedirs("../reports", exist_ok=True)
    with open("../reports/step2_metrics.json", "w") as f:
        json.dump(report, f, indent=4)
    print(f"Saved metrics to ../reports/step2_metrics.json")