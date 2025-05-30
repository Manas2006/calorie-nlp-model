"""Utility functions for text processing and feature extraction."""
import re
from typing import Optional

def clean_text(text: str) -> str:
    """Clean text by removing punctuation and converting to lowercase.
    
    Args:
        text: Input text to clean
        
    Returns:
        Cleaned text with only lowercase letters and spaces
    """
    return re.sub(r"[^a-z\s]", "", text.lower()).strip()

def token_count(text: str) -> int:
    """Count the number of tokens in a text.
    
    Args:
        text: Input text
        
    Returns:
        Number of space-separated tokens
    """
    return len(text.split())

def extract_serving_size(serving: Optional[str]) -> Optional[float]:
    """Extract serving size in grams from serving text.
    
    Args:
        serving: Text containing serving size information
        
    Returns:
        Serving size in grams if found, None otherwise
    """
    if not serving:
        return None
    
    # Look for patterns like "100 g" or "100g"
    match = re.search(r'(\d+(?:\.\d+)?)\s*g', serving)
    if match:
        return float(match.group(1))
    return None 