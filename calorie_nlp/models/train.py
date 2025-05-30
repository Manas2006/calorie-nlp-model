"""Model training and evaluation functionality."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import os
from datetime import datetime
from sklearn.model_selection import KFold
from ..data.datasets import create_calorie_bins
from .mlp import CalorieMLP

class Trainer:
    """Trainer class for the CalorieMLP model."""
    
    def __init__(self, config: Dict):
        """Initialize the trainer.
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create experiment directory
        self.exp_dir = os.path.join(
            "experiments",
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Initialize log file
        self.log_file = os.path.join(self.exp_dir, "training.log")
        with open(self.log_file, "w") as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write(f"Config: {config}\n\n")
    
    def _log(self, message: str):
        """Log a message to the log file.
        
        Args:
            message: Message to log
        """
        with open(self.log_file, "a") as f:
            f.write(f"{message}\n")
        print(message)
    
    def _create_dataloader(
        self,
        embeddings: np.ndarray,
        token_counts: np.ndarray,
        log_calories: np.ndarray,
        batch_size: int,
        shuffle: bool = True
    ) -> DataLoader:
        """Create a DataLoader for the given data.
        
        Args:
            embeddings: Array of embeddings
            token_counts: Array of token counts
            log_calories: Array of log calories
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for the data
        """
        # Combine embeddings and token counts
        features = np.concatenate([
            embeddings,
            token_counts.reshape(-1, 1)
        ], axis=1)
        
        # Create dataset
        dataset = TensorDataset(
            torch.FloatTensor(features),
            torch.FloatTensor(log_calories)
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )
    
    def _train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Train for one epoch.
        
        Args:
            model: Model to train
            train_loader: DataLoader for training data
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            Average training loss
        """
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / len(train_loader)
    
    def _evaluate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """Evaluate the model.
        
        Args:
            model: Model to evaluate
            val_loader: DataLoader for validation data
            criterion: Loss function
            
        Returns:
            Average validation loss
        """
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch_x, batch_y in pbar:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / len(val_loader)
    
    def fit(
        self,
        embeddings: np.ndarray,
        token_counts: np.ndarray,
        log_calories: np.ndarray,
        calorie_bins: np.ndarray
    ) -> Tuple[nn.Module, List[float]]:
        """Train the model using k-fold cross-validation.
        
        Args:
            embeddings: Array of embeddings
            token_counts: Array of token counts
            log_calories: Array of log calories
            calorie_bins: Array of calorie bins for stratification
            
        Returns:
            Tuple of (best model, list of fold MAEs)
        """
        kf = KFold(
            n_splits=self.config["k_folds"],
            shuffle=True,
            random_state=42
        )
        
        fold_maes = []
        best_mae = float("inf")
        best_model = None
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(embeddings, calorie_bins)):
            self._log(f"\nFold {fold + 1}/{self.config['k_folds']}")
            
            # Create data loaders
            train_loader = self._create_dataloader(
                embeddings[train_idx],
                token_counts[train_idx],
                log_calories[train_idx],
                self.config["batch_size"]
            )
            
            val_loader = self._create_dataloader(
                embeddings[val_idx],
                token_counts[val_idx],
                log_calories[val_idx],
                self.config["batch_size"],
                shuffle=False
            )
            
            # Initialize model and training components
            model = CalorieMLP().to(self.device)
            criterion = nn.HuberLoss(delta=self.config["delta"])
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config["lr"]
            )
            
            # Training loop
            best_val_loss = float("inf")
            patience_counter = 0
            
            for epoch in range(self.config["epochs"]):
                train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
                val_loss = self._evaluate(model, val_loader, criterion)
                
                self._log(
                    f"Epoch {epoch + 1}/{self.config['epochs']}, "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model for this fold
                    torch.save(
                        model.state_dict(),
                        os.path.join(self.exp_dir, f"best_model_fold_{fold}.pth")
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= self.config["patience"]:
                        self._log(f"Early stopping triggered after {epoch + 1} epochs")
                        break
            
            # Evaluate on validation set
            model.load_state_dict(
                torch.load(os.path.join(self.exp_dir, f"best_model_fold_{fold}.pth"))
            )
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    outputs = model(batch_x)
                    val_preds.extend(outputs.cpu().numpy())
                    val_true.extend(batch_y.numpy())
            
            # Calculate MAE on original scale
            mae = np.mean(np.abs(
                np.expm1(val_preds) - np.expm1(val_true)
            ))
            fold_maes.append(mae)
            
            self._log(f"Fold {fold + 1} MAE: {mae:.2f} calories per 100g")
            
            # Update best model
            if mae < best_mae:
                best_mae = mae
                best_model = model
                # Save best model overall
                torch.save(
                    model.state_dict(),
                    os.path.join("models", "best_mlp.pth")
                )
        
        # Log final results
        self._log(f"\nFinal Results:")
        self._log(f"Fold MAEs: {[f'{mae:.2f}' for mae in fold_maes]}")
        self._log(f"Mean MAE: {np.mean(fold_maes):.2f} calories per 100g")
        self._log(f"Std MAE: {np.std(fold_maes):.2f} calories per 100g")
        
        return best_model, fold_maes 