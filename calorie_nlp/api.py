"""FastAPI server for calorie predictions."""
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from calorie_nlp.models.mlp import CalorieMLP
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Calorie Prediction API",
    description="API for predicting calories in food items using NLP",
    version="1.0.0"
)

# Global variables for model and embedder
model = None
embedder = None

class FoodItem(BaseModel):
    """Food item for prediction."""
    name: str

class BatchFoodItems(BaseModel):
    """Batch of food items for prediction."""
    items: List[str]

class PredictionResponse(BaseModel):
    """Response model for predictions."""
    food: str
    calories_per_100g: float

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]

def load_models():
    """Load the MLP model and sentence embedder."""
    global model, embedder
    
    try:
        # Load models with memory optimization
        torch.set_num_threads(1)  # Limit CPU threads
        torch.set_num_interop_threads(1)
        
        # Load embedder with memory optimization
        embedder = SentenceTransformer('all-mpnet-base-v2')
        embedder.max_seq_length = 128  # Reduce max sequence length
        
        # Load MLP model
        model = CalorieMLP()
        model.eval()  # Set to evaluation mode
        
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    load_models()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_calories(food: FoodItem):
    """Predict calories for a single food item."""
    try:
        # Get embedding
        embedding = embedder.encode(food.name, convert_to_tensor=True)
        
        # Add token count feature
        token_count = len(food.name.split())
        features = torch.cat([embedding, torch.tensor([token_count], dtype=torch.float32)])
        
        # Make prediction
        with torch.no_grad():
            calories = model(features.unsqueeze(0))
        
        return {
            "food": food.name,
            "calories_per_100g": float(calories.item())
        }
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_calories_batch(foods: BatchFoodItems):
    """Predict calories for multiple food items."""
    try:
        predictions = []
        for food_name in foods.items:
            # Get embedding
            embedding = embedder.encode(food_name, convert_to_tensor=True)
            
            # Add token count feature
            token_count = len(food_name.split())
            features = torch.cat([embedding, torch.tensor([token_count], dtype=torch.float32)])
            
            # Make prediction
            with torch.no_grad():
                calories = model(features.unsqueeze(0))
            
            predictions.append({
                "food": food_name,
                "calories_per_100g": float(calories.item())
            })
        
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Error making batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 