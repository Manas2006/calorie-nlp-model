"""FastAPI server for calorie predictions."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from .predict import load_model_and_embedder, predict_calories

app = FastAPI(
    title="Calorie Prediction API",
    description="API for predicting calories in food items using NLP",
    version="1.0.0"
)

# Load model and embedder at startup
print("Loading model and embedder...")
model, embedder = load_model_and_embedder()

class FoodItem(BaseModel):
    """Food item for prediction."""
    name: str

class FoodItems(BaseModel):
    """Multiple food items for prediction."""
    items: List[str]

class Prediction(BaseModel):
    """Prediction response."""
    food: str
    calories_per_100g: float

class Predictions(BaseModel):
    """Multiple predictions response."""
    predictions: List[Prediction]

@app.post("/predict", response_model=Prediction)
async def predict_single(food: FoodItem):
    """Predict calories for a single food item."""
    try:
        pred = predict_calories([food.name], model, embedder)[0]
        return Prediction(food=food.name, calories_per_100g=pred)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=Predictions)
async def predict_batch(foods: FoodItems):
    """Predict calories for multiple food items."""
    try:
        preds = predict_calories(foods.items, model, embedder)
        predictions = [
            Prediction(food=food, calories_per_100g=pred)
            for food, pred in zip(foods.items, preds)
        ]
        return Predictions(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

def start():
    """Start the FastAPI server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start() 