"""Unit tests for the MLP model."""
import torch
import pytest
from calorie_nlp.models.mlp import CalorieMLP

def test_mlp_forward():
    """Test forward pass of the MLP model."""
    batch_size = 4
    input_dim = 769  # 768 (mpnet) + 1 (token_count)
    
    # Create model and input
    model = CalorieMLP()
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size,)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

def test_mlp_dropout():
    """Test that dropout is applied during training but not during evaluation."""
    model = CalorieMLP()
    # Use batch size of 2 to avoid BatchNorm error
    x = torch.randn(2, 769)
    
    # Training mode
    model.train()
    output_train = model(x)
    
    # Evaluation mode
    model.eval()
    output_eval = model(x)
    
    # Outputs should be different in training mode due to dropout
    assert not torch.allclose(output_train, output_eval) 