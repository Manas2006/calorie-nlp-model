"""Main training script for the calorie prediction model."""
import os
import tomli
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from .data.datasets import get_combined_df, create_calorie_bins
from .models.train import Trainer

def main():
    # Load configuration
    with open("config.toml", "rb") as f:
        config = tomli.load(f)
    
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    os.makedirs(config["embedding"]["cache_dir"], exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = get_combined_df(
        config["data"]["recipes_path"],
        config["data"]["fruits_path"]
    )
    print(f"Loaded {len(df)} food items")
    
    # Create calorie bins for stratification
    calorie_bins = create_calorie_bins(
        df,
        n_bins=config["data"]["n_bins"]
    )
    
    # Load embedder and get embeddings
    print("Loading embedder and generating embeddings...")
    embedder = SentenceTransformer(
        config["embedding"]["model_name"],
        cache_folder=config["embedding"]["cache_dir"]
    )
    
    # Generate embeddings for all food names with progress bar
    print("Generating embeddings (this may take a few minutes)...")
    embeddings = []
    batch_size = 32
    for i in tqdm(range(0, len(df), batch_size)):
        batch = df["Name"].tolist()[i:i + batch_size]
        batch_embeddings = embedder.encode(batch)
        embeddings.extend(batch_embeddings)
    embeddings = np.array(embeddings)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    
    # Initialize trainer
    trainer = Trainer(config["train"])
    
    # Train model
    print("\nStarting training...")
    best_model, fold_maes = trainer.fit(
        embeddings=embeddings,
        token_counts=df["token_count"].values,
        log_calories=df["log_calories"].values,
        calorie_bins=calorie_bins
    )
    
    print("\nTraining complete!")
    print(f"Best model saved to: models/best_mlp.pth")
    print(f"Training logs saved to: {trainer.exp_dir}")

if __name__ == "__main__":
    main() 