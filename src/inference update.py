# src/inference_api.py

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Correctly import from another file within the `src` directory
from src.nlp_predict import model_inference_engine

app = FastAPI(
    title="Operation CIIS - Inference API",
    description="API for classifying tweets using the fine-tuned multilingual detector.",
    version="1.0.0"
)

# Pydantic models for request/response validation
class InferenceRequest(BaseModel):
    texts: List[str]

class PredictionResult(BaseModel):
    label: str
    score: float

class InferenceResponse(BaseModel):
    predictions: List[PredictionResult]

@app.post("/predict", response_model=InferenceResponse, tags=["Machine Learning"])
async def run_inference(request: InferenceRequest):
    """Receives a list of texts and returns classification labels and scores."""
    if not request.texts:
        raise HTTPException(status_code=400, detail="Input text list cannot be empty.")
    
    predictions = model_inference_engine.predict(request.texts)
    return {"predictions": predictions}

@app.get("/health", tags=["System"])
async def health_check():
    """Provides a simple health check of the API."""
    return {"status": "ok", "model_loaded": model_inference_engine is not None}

# To run the API server, execute this from the project's ROOT directory:
# uvicorn src.inference_api:app --reload --host 0.0.0.0 --port 8000