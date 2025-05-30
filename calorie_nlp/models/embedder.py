"""SentenceTransformer model loading and caching."""
from sentence_transformers import SentenceTransformer
from joblib import load, dump
import pathlib
from typing import Optional

def load_embedder(cache_dir: str = "model_cache") -> SentenceTransformer:
    """Load or create and cache the SentenceTransformer model.
    
    Args:
        cache_dir: Directory to store the cached model
        
    Returns:
        Loaded SentenceTransformer model
    """
    path = pathlib.Path(cache_dir, "mpnet.pkl")
    if path.exists():
        return load(path)
    
    model = SentenceTransformer("all-mpnet-base-v2", cache_folder=cache_dir)
    dump(model, path)
    return model 