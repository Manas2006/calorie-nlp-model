"""MLP model for calorie prediction."""
import torch
import torch.nn as nn

class CalorieMLP(nn.Module):
    """MLP model for predicting log calories from embeddings and features."""
    
    def __init__(self, input_dim: int = 769):  # 768 (mpnet) + 1 (token_count)
        """Initialize the MLP model.
        
        Args:
            input_dim: Input dimension (embedding size + feature size)
        """
        super().__init__()
        
        self.layers = nn.Sequential(
            # First layer
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second layer
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Third layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output layer
            nn.Linear(128, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Predicted log calories of shape (batch_size, 1)
        """
        return self.layers(x).squeeze() 