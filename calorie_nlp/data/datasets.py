"""Dataset loading and preprocessing functions."""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from .utils import clean_text, token_count, extract_serving_size

def load_recipes(path: str) -> pd.DataFrame:
    """Load and preprocess the recipes dataset.
    
    Args:
        path: Path to recipes.csv
        
    Returns:
        DataFrame with cleaned and processed recipe data
    """
    df = pd.read_csv(path)
    
    # If recipes.csv doesn't have serving size, assume 100g servings
    if "Serving" not in df.columns:
        df["Serving"] = "100 g"
    
    df = df[["Name", "Serving", "Calories"]].dropna()
    df["Name"] = df["Name"].apply(clean_text)
    df["token_count"] = df["Name"].apply(token_count)
    
    # Standardize calories to per 100g
    df["Calories_per_100g"] = df.apply(
        lambda row: (row["Calories"] / extract_serving_size(row["Serving"]) * 100)
        if extract_serving_size(row["Serving"])
        else None,
        axis=1
    )
    
    # Add log calories
    df["log_calories"] = np.log1p(df["Calories_per_100g"])
    
    return df[["Name", "token_count", "Calories_per_100g", "log_calories"]].dropna()

def load_fruits(path: str) -> pd.DataFrame:
    """Load and preprocess the fruits dataset.
    
    Args:
        path: Path to fruits.csv
        
    Returns:
        DataFrame with cleaned and processed fruit data
    """
    df = pd.read_csv(path)
    
    # Extract numeric calories from the "Calories" column
    df["Calories"] = df["Calories"].str.extract(r'(\d+)').astype(float)
    df = df[["Food", "Serving", "Calories"]].dropna()
    df = df.rename(columns={"Food": "Name"})
    
    df["Name"] = df["Name"].apply(clean_text)
    df["token_count"] = df["Name"].apply(token_count)
    
    # Standardize calories to per 100g
    df["Calories_per_100g"] = df.apply(
        lambda row: (row["Calories"] / extract_serving_size(row["Serving"]) * 100)
        if extract_serving_size(row["Serving"])
        else None,
        axis=1
    )
    
    # Add log calories
    df["log_calories"] = np.log1p(df["Calories_per_100g"])
    
    return df[["Name", "token_count", "Calories_per_100g", "log_calories"]].dropna()

def get_combined_df(recipes_path: str, fruits_path: str) -> pd.DataFrame:
    """Combine and preprocess both datasets.
    
    Args:
        recipes_path: Path to recipes.csv
        fruits_path: Path to fruits.csv
        
    Returns:
        Combined DataFrame with cleaned and processed data
    """
    recipes_df = load_recipes(recipes_path)
    fruits_df = load_fruits(fruits_path)
    
    # Combine datasets
    df = pd.concat([recipes_df, fruits_df], ignore_index=True)
    
    # Remove extreme outliers
    df = df[(df["Calories_per_100g"] > 0) & (df["Calories_per_100g"] < 1000)]
    
    return df

def create_calorie_bins(df: pd.DataFrame, n_bins: int = 5) -> np.ndarray:
    """Create stratified bins based on calorie values.
    
    Args:
        df: DataFrame with Calories_per_100g column
        n_bins: Number of bins to create
        
    Returns:
        Array of bin labels
    """
    return pd.qcut(df["Calories_per_100g"], n_bins, labels=False).values 