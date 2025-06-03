"""Script for making calorie predictions using the trained model."""
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from .models.mlp import CalorieMLP
from .data.utils import clean_text, token_count
import tomli
import argparse

def load_model_and_embedder(config_path: str = "config.toml"):
    """Load the trained model and embedder.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Tuple of (model, embedder)
    """
    # Load config
    with open(config_path, "rb") as f:
        config = tomli.load(f)
    
    # Load embedder
    embedder = SentenceTransformer(
        config["embedding"]["model_name"],
        cache_folder=config["embedding"]["cache_dir"]
    )
    
    # Initialize and load model
    model = CalorieMLP()
    model.load_state_dict(torch.load("models/best_mlp.pth"))
    model.eval()
    
    return model, embedder

def predict_calories(
    food_names: list[str],
    model: CalorieMLP,
    embedder: SentenceTransformer
) -> list[float]:
    """Predict calories for a list of food names.
    
    Args:
        food_names: List of food names to predict
        model: Trained model
        embedder: Sentence transformer model
        
    Returns:
        List of predicted calories per 100g
    """
    # Clean and prepare inputs
    cleaned_names = [clean_text(name) for name in food_names]
    token_counts = np.array([token_count(name) for name in cleaned_names])
    
    # Generate embeddings
    embeddings = embedder.encode(cleaned_names)
    
    # Combine features
    features = np.concatenate([
        embeddings,
        token_counts.reshape(-1, 1)
    ], axis=1)
    
    # Make predictions
    with torch.no_grad():
        inputs = torch.FloatTensor(features)
        log_preds = model(inputs)
        preds = torch.expm1(log_preds).numpy()
        
        # Handle both single and multiple predictions
        if preds.ndim == 0:
            preds = np.array([preds])
    
    return preds.tolist()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Predict calories for food items")
    parser.add_argument("--input", type=str, help="Food item to predict calories for")
    parser.add_argument("--model-path", type=str, default="models/best_mlp.pth", help="Path to the model checkpoint")
    args = parser.parse_args()
    
    # Load model and embedder
    print("Loading model and embedder...")
    model, embedder = load_model_and_embedder()
    
    # Make predictions
    print("\nMaking predictions...")
    if args.input:
        # Single food prediction
        pred = predict_calories([args.input], model, embedder)[0]
        print("\nPrediction:")
        print("-" * 50)
        print(f"{'Food Item':<30} {'Calories/100g':>15}")
        print("-" * 50)
        print(f"{args.input:<30} {pred:>15.1f}")
        print("-" * 50)
    else:
        # Test foods (default behavior)
        test_foods = [
            "apple", "banana", "chicken breast",
            "grilled chicken breast salad", "steamed vegetables", "baked salmon",
            "deep fried chicken wings", "creamy chocolate cake", "extra cheesy pizza",
            "chicken curry with rice", "beef stir fry with noodles", "vegetable pasta",
            "chocolate ice cream", "fruit salad", "cheesecake",
            "potato chips", "trail mix", "protein bar",
            "paneer tikka masala", "paneer bhurji", "zero calorie diet coke"
        ]
        predictions = predict_calories(test_foods, model, embedder)
        print("\nPredictions:")
        print("-" * 50)
        print(f"{'Food Item':<30} {'Calories/100g':>15}")
        print("-" * 50)
        for food, pred in zip(test_foods, predictions):
            print(f"{food:<30} {pred:>15.1f}")
        print("-" * 50)

if __name__ == "__main__":
    main() 