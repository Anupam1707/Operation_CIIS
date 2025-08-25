# src/nlp_predict.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This path now points to our new multilingual model location
MODEL_PATH = "models/multilingual_detector/"

class ModelInference:
    """A singleton class to handle loading the model and running inference."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelInference, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path: str = MODEL_PATH):
        # The __init__ is only run the first time
        if hasattr(self, 'model'):
            return

        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()  # Set model to evaluation mode
            logger.info(f"✅ Model loaded successfully from '{model_path}' on device '{self.device}'.")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise

    def predict(self, texts: List[str]) -> List[Dict]:
        """Classify a batch of tweet texts."""
        if not texts or not all(isinstance(t, str) for t in texts):
            logger.warning("Prediction received empty or invalid input.")
            return []
            
        try:
            inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=280, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            results = []
            for pred in predictions:
                score = pred.max().item()
                label_id = pred.argmax().item()
                label = self.model.config.id2label[label_id]
                results.append({"label": label, "score": round(score, 4)})
            
            return results
        except Exception as e:
            logger.error(f"❌ Error during prediction: {e}")
            return [{"label": "error", "score": 0.0}] * len(texts)

# Create a single, pre-loaded instance of the model to be imported by other scripts.
# This prevents reloading the heavy model into memory multiple times.
model_inference_engine = ModelInference()