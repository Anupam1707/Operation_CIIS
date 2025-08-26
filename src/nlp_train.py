
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

# --- Data Loading and Preprocessing ---

# Load synthetic data
with open("data/synthetic_posts.json", "r") as f:
    synthetic_data = json.load(f)

# Convert to a list of dictionaries with 'text' and 'label' keys
processed_synthetic_data = []
for post in synthetic_data:
    processed_synthetic_data.append({
        "text": post["data"]["text"],
        "label": 1 if post["label"] == "anti-indian" else 0
    })

synthetic_df = pd.DataFrame(processed_synthetic_data)

# Load real data if it exists
real_data_path = "data/dataset_processed.csv"
if os.path.exists(real_data_path):
    real_df = pd.read_csv(real_data_path)
    # Assuming real_df has 'text' and 'label' columns
    combined_df = pd.concat([synthetic_df, real_df], ignore_index=True)
else:
    combined_df = synthetic_df

dataset = Dataset.from_pandas(combined_df)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Split into train and test
train_test = dataset.train_test_split(test_size=0.2)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]

# --- Model and Metrics ---

# Load model
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Compute metrics
def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# --- Training ---

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

if __name__ == "__main__":
    # Train the model
    trainer.train()

    # Evaluate and log metrics
    metrics = trainer.evaluate()
    print("Evaluation Metrics:", metrics)

    # Save trained model
    model_path = "models/multilingual_detector"
    os.makedirs(model_path, exist_ok=True)
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Saved model to {model_path}")

    # Generate evaluation report
    report_path = "reports"
    os.makedirs(report_path, exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "overall": {
                "accuracy": metrics.get("eval_accuracy"),
                "precision": metrics.get("eval_precision"),
                "recall": metrics.get("eval_recall"),
                "f1": metrics.get("eval_f1")
            }
        }
    }
    with open(f"{report_path}/step2_metrics.json", "w") as f:
        json.dump(report, f, indent=4)
    print(f"Saved metrics to {report_path}/step2_metrics.json")
