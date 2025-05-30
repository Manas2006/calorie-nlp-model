"""Unit tests for utility functions."""
import pytest
from calorie_nlp.data.utils import clean_text, token_count, extract_serving_size

def test_clean_text():
    """Test text cleaning function."""
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("Grilled Chicken (100g)") == "grilled chicken g"
    assert clean_text("") == ""

def test_token_count():
    """Test token counting function."""
    assert token_count("hello world") == 2
    assert token_count("grilled chicken breast") == 3
    assert token_count("") == 0

def test_extract_serving_size():
    """Test serving size extraction function."""
    assert extract_serving_size("100 g") == 100.0
    assert extract_serving_size("50g") == 50.0
    assert extract_serving_size("1.5 g") == 1.5
    assert extract_serving_size("No serving size") is None
    assert extract_serving_size(None) is None 