"""Tests for the MLP model."""
import torch
from calorie_nlp.models.mlp import CalorieMLP

def test_model_initialization():
    """Test model initialization."""
    model = CalorieMLP()
    assert isinstance(model, CalorieMLP)
    assert model.layers is not None

def test_model_forward_pass():
    """Test model forward pass."""
    model = CalorieMLP()
    batch_size = 4
    input_dim = 769  # 768 (mpnet) + 1 (token_count)
    
    # Create random input tensor
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    output = model(x)
    
    # Check output shape and type
    assert output.shape == (batch_size,)
    assert isinstance(output, torch.Tensor)

def test_model_layers():
    """Test model architecture."""
    model = CalorieMLP()
    
    # Check number of layers
    assert len(model.layers) == 12  # 4 layer blocks with 3 components each
    
    # Check layer types
    assert isinstance(model.layers[0], torch.nn.Linear)
    assert isinstance(model.layers[1], torch.nn.BatchNorm1d)
    assert isinstance(model.layers[2], torch.nn.ReLU)
    assert isinstance(model.layers[3], torch.nn.Dropout)

def test_model_dropout():
    """Test dropout behavior."""
    model = CalorieMLP()
    model.train()  # Enable dropout
    
    batch_size = 4
    input_dim = 769
    x = torch.randn(batch_size, input_dim)
    
    # Two forward passes should give different results due to dropout
    output1 = model(x)
    output2 = model(x)
    assert not torch.allclose(output1, output2)

def test_model_eval_mode():
    """Test model in evaluation mode."""
    model = CalorieMLP()
    model.eval()  # Disable dropout
    
    batch_size = 4
    input_dim = 769
    x = torch.randn(batch_size, input_dim)
    
    # Two forward passes should give same results in eval mode
    with torch.no_grad():
        output1 = model(x)
        output2 = model(x)
    assert torch.allclose(output1, output2) 