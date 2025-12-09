"""
Training Module for Neural Network Melody Models
Provides training infrastructure with learning curves, validation, and checkpointing.
"""

import logging
import time
import random
from typing import List, Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = None

try:
    from .data_loader import NottinghamDataLoader, split_dataset
    HAS_DATA_LOADER = True
except ImportError:
    HAS_DATA_LOADER = False


class MelodyDataset:
    """Dataset for melody sequences."""
    
    def __init__(self, sequences: List[List[int]], sequence_length: int = 16):
        """
        Initialize melody dataset.
        
        Args:
            sequences: List of melody sequences (as note indices)
            sequence_length: Length of input sequences
        """
        self.sequences = sequences
        self.sequence_length = sequence_length
        self.data = self._prepare_sequences()
    
    def _prepare_sequences(self) -> List[Tuple[List[int], int]]:
        """Prepare input-output pairs from sequences."""
        data = []
        
        for sequence in self.sequences:
            # Create sliding windows
            for i in range(len(sequence) - self.sequence_length):
                input_seq = sequence[i:i + self.sequence_length]
                target = sequence[i + self.sequence_length]
                data.append((input_seq, target))
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        return self.data[idx]


class TrainingHistory:
    """Tracks and visualizes training history."""
    
    def __init__(self):
        """Initialize training history tracker."""
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epochs = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def update(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float
    ):
        """Update history with new metrics."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accuracies.append(train_acc)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        
        # Track best validation loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
    
    def plot_curves(self, save_path: Optional[Path] = None, model_name: str = "model"):
        """
        Plot training and validation curves.
        
        Args:
            save_path: Optional path to save the plot
            model_name: Name of the model for description file
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Loss curves
        axes[0, 0].plot(self.epochs, self.train_losses, label='Train Loss', marker='o', linewidth=2)
        axes[0, 0].plot(self.epochs, self.val_losses, label='Val Loss', marker='s', linewidth=2)
        axes[0, 0].axvline(x=self.best_epoch, color='r', linestyle='--', 
                          label=f'Best Epoch ({self.best_epoch})')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Loss Curves', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[0, 1].plot(self.epochs, self.train_accuracies, label='Train Acc', marker='o', linewidth=2)
        axes[0, 1].plot(self.val_accuracies, label='Val Acc', marker='s', linewidth=2)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate schedule
        axes[1, 0].plot(self.epochs, self.learning_rates, marker='o', color='green', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Overfitting analysis
        generalization_gap = np.array(self.train_losses) - np.array(self.val_losses)
        axes[1, 1].plot(self.epochs, generalization_gap, marker='o', color='purple', linewidth=2)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Train Loss - Val Loss', fontsize=12)
        axes[1, 1].set_title('Generalization Gap', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            # Ensure graphs directory exists
            save_path = Path(save_path)
            graphs_dir = save_path.parent / "graphs"
            graphs_dir.mkdir(parents=True, exist_ok=True)
            
            # Save plot in graphs folder
            graph_path = graphs_dir / save_path.name
            plt.savefig(graph_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {graph_path}")
            
            # Generate description file
            desc_path = graph_path.with_suffix('.txt')
            self._save_description(desc_path, model_name)
            print(f"Graph description saved to {desc_path}")
        else:
            plt.show()
    
    def _save_description(self, path: Path, model_name: str):
        """Generate and save description file for training curves."""
        # Calculate final metrics
        final_train_loss = self.train_losses[-1] if self.train_losses else 0
        final_val_loss = self.val_losses[-1] if self.val_losses else 0
        final_train_acc = self.train_accuracies[-1] if self.train_accuracies else 0
        final_val_acc = self.val_accuracies[-1] if self.val_accuracies else 0
        
        # Calculate improvement
        if len(self.train_losses) > 1:
            train_loss_improvement = self.train_losses[0] - final_train_loss
            val_loss_improvement = self.val_losses[0] - final_val_loss
        else:
            train_loss_improvement = 0
            val_loss_improvement = 0
        
        # Assess overfitting
        generalization_gap = final_train_loss - final_val_loss
        if generalization_gap < 0.05:
            overfitting_status = "No overfitting detected - model generalizes well"
        elif generalization_gap < 0.15:
            overfitting_status = "Minimal overfitting - acceptable generalization"
        elif generalization_gap < 0.3:
            overfitting_status = "Moderate overfitting - consider regularization"
        else:
            overfitting_status = "Significant overfitting - model memorizing training data"
        
        # Assess convergence
        if len(self.val_losses) >= 5:
            recent_val_change = abs(self.val_losses[-1] - self.val_losses[-5])
            if recent_val_change < 0.01:
                convergence_status = "Model has converged - minimal improvement in recent epochs"
            elif recent_val_change < 0.05:
                convergence_status = "Model is converging - slow improvement"
            else:
                convergence_status = "Model still improving - continue training"
        else:
            convergence_status = "Not enough epochs to assess convergence"
        
        description = f"""TRAINING CURVES ANALYSIS - {model_name}
{'=' * 80}

GRAPH OVERVIEW:
This graph shows the training history across {len(self.epochs)} epochs.
It contains 4 subplots that visualize different aspects of the training process.

{'=' * 80}
SUBPLOT 1: LOSS CURVES (Top Left)
{'=' * 80}

What it shows:
- Blue line: Training loss over epochs
- Orange line: Validation loss over epochs  
- Red dashed line: Best epoch (epoch {self.best_epoch})

Key Metrics:
- Initial Train Loss: {self.train_losses[0]:.4f}
- Final Train Loss: {final_train_loss:.4f}
- Train Loss Improvement: {train_loss_improvement:.4f}

- Initial Val Loss: {self.val_losses[0]:.4f}
- Final Val Loss: {final_val_loss:.4f}
- Val Loss Improvement: {val_loss_improvement:.4f}

- Best Val Loss: {self.best_val_loss:.4f} (at epoch {self.best_epoch})

Interpretation:
✓ Both losses should trend downward (indicates learning)
✓ Losses converging indicates model is learning the patterns
✗ If validation loss increases while training decreases → overfitting

Current Status: {overfitting_status}

{'=' * 80}
SUBPLOT 2: ACCURACY CURVES (Top Right)
{'=' * 80}

What it shows:
- Training and validation accuracy over time
- Higher is better (range: 0.0 to 1.0)

Key Metrics:
- Final Train Accuracy: {final_train_acc:.2%}
- Final Val Accuracy: {final_val_acc:.2%}
- Accuracy Gap: {abs(final_train_acc - final_val_acc):.2%}

Interpretation:
✓ Both accuracies should increase over time
✓ Gap < 5% indicates good generalization
✗ Large gap indicates overfitting (model memorizing training data)

{'=' * 80}
SUBPLOT 3: LEARNING RATE SCHEDULE (Bottom Left)
{'=' * 80}

What it shows:
- Learning rate changes during training (log scale)
- Shows if/when learning rate was reduced

Key Metrics:
- Initial LR: {self.learning_rates[0]:.6f}
- Final LR: {self.learning_rates[-1]:.6f}

Interpretation:
✓ Decreasing LR helps model fine-tune in later epochs
✓ Plateau indicates stable learning rate
✗ If LR drops too early, model may converge prematurely

{'=' * 80}
SUBPLOT 4: GENERALIZATION GAP (Bottom Right)
{'=' * 80}

What it shows:
- Difference between training and validation loss
- Purple line crossing zero = overfitting point

Key Metrics:
- Current Gap: {generalization_gap:.4f}

Interpretation:
✓ Gap near 0: Model generalizes well
✓ Slightly negative: Validation better than training (ideal)
✗ Large positive gap: Overfitting (model memorizes training data)
✗ Increasing gap over time: Progressive overfitting

Current Generalization: {generalization_gap:.4f}

{'=' * 80}
OVERALL ASSESSMENT
{'=' * 80}

Convergence: {convergence_status}

Overfitting: {overfitting_status}

Recommendations:
"""
        
        # Add specific recommendations
        recommendations = []
        
        if generalization_gap > 0.3:
            recommendations.append("• Increase dropout rate or add regularization")
            recommendations.append("• Reduce model complexity (fewer layers/units)")
            recommendations.append("• Collect more training data")
        
        if final_val_acc < 0.5:
            recommendations.append("• Model is underperforming - consider different architecture")
            recommendations.append("• Try increasing model capacity or training longer")
        
        if len(self.val_losses) >= 5 and abs(self.val_losses[-1] - self.val_losses[-5]) < 0.01:
            recommendations.append("• Training has plateaued - can stop training")
            recommendations.append("• Use early stopping to save time in future runs")
        
        if not recommendations:
            recommendations.append("• Training appears healthy - continue current approach")
            recommendations.append("• Monitor performance on test set for final evaluation")
        
        description += "\n".join(recommendations)
        
        description += f"""\n
{'=' * 80}
NEXT STEPS
{'=' * 80}

1. Review the best model checkpoint from epoch {self.best_epoch}
2. Run comprehensive testing on held-out test set
3. Compare with other model architectures
4. Evaluate generated melodies qualitatively
5. Consider hyperparameter tuning if performance is suboptimal

{'=' * 80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
"""
        
        with open(path, 'w') as f:
            f.write(description)
    
    def save(self, path: Path):
        """Save training history to JSON."""
        data = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'TrainingHistory':
        """Load training history from JSON."""
        history = cls()
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        history.epochs = data['epochs']
        history.train_losses = data['train_losses']
        history.val_losses = data['val_losses']
        history.train_accuracies = data['train_accuracies']
        history.val_accuracies = data['val_accuracies']
        history.learning_rates = data['learning_rates']
        history.best_val_loss = data['best_val_loss']
        history.best_epoch = data['best_epoch']
        
        return history


class ModelTrainer:
    """Trainer for neural network melody models."""
    
    def __init__(
        self,
        model: nn.Module,
        model_name: str = "melody_model",
        device: Optional[str] = None
    ):
        """
        Initialize model trainer.
        
        Args:
            model: PyTorch model to train
            model_name: Name for saving checkpoints
            device: Device to train on ('cuda' or 'cpu')
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for model training")
        
        self.model = model
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.logger = logging.getLogger(__name__)
        self.history = TrainingHistory()
    
    def train(
        self,
        train_data: List[List[int]],
        val_data: List[List[int]],
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        sequence_length: int = 16,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[Path] = None,
        lr_scheduler: Optional[str] = 'reduce_on_plateau',
        **kwargs
    ) -> TrainingHistory:
        """
        Train the model with validation and learning curves.
        
        Args:
            train_data: Training sequences
            val_data: Validation sequences
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Initial learning rate
            sequence_length: Length of input sequences
            early_stopping_patience: Epochs to wait before early stopping
            checkpoint_dir: Directory to save checkpoints
            lr_scheduler: Learning rate scheduler type
            
        Returns:
            Training history
        """
        self.logger.info(f"Training {self.model_name} on {self.device}")
        self.logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
        
        # Create datasets
        train_dataset = MelodyDataset(train_data, sequence_length)
        val_dataset = MelodyDataset(val_data, sequence_length)
        
        # Create data loaders
        train_loader = self._create_dataloader(train_dataset, batch_size, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, batch_size, shuffle=False)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Setup learning rate scheduler
        scheduler = self._create_scheduler(optimizer, lr_scheduler)
        
        # Checkpoint directory
        if checkpoint_dir:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Train one epoch
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer
            )
            
            # Validate
            val_loss, val_acc = self._validate(val_loader, criterion)
            
            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler:
                if lr_scheduler == 'reduce_on_plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Update history
            self.history.update(epoch, train_loss, val_loss, train_acc, val_acc, current_lr)
            
            # Logging
            epoch_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch}/{epochs} ({epoch_time:.2f}s) - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%} | "
                f"LR: {current_lr:.6f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                if checkpoint_dir:
                    self._save_checkpoint(
                        checkpoint_dir / f"{self.model_name}_best.pt",
                        epoch, optimizer
                    )
                    self.logger.info(f"✓ New best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs")
                break
            
            # Save periodic checkpoint
            if checkpoint_dir and epoch % 10 == 0:
                self._save_checkpoint(
                    checkpoint_dir / f"{self.model_name}_epoch_{epoch}.pt",
                    epoch, optimizer
                )
        
        # Save final model and history
        if checkpoint_dir:
            self._save_checkpoint(
                checkpoint_dir / f"{self.model_name}_final.pt",
                epoch, optimizer
            )
            self.history.save(checkpoint_dir / f"{self.model_name}_history.json")
            
            # Plot and save learning curves
            self.history.plot_curves(
                checkpoint_dir / f"{self.model_name}_curves.png",
                model_name=self.model_name
            )
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {self.history.best_val_loss:.4f} at epoch {self.history.best_epoch}")
        
        return self.history
    
    def _train_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _create_dataloader(
        self,
        dataset: MelodyDataset,
        batch_size: int,
        shuffle: bool
    ) -> DataLoader:
        """Create a DataLoader from dataset."""
        
        class TorchDataset(torch.utils.data.Dataset):
            def __init__(self, melody_dataset):
                self.dataset = melody_dataset
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                inputs, target = self.dataset[idx]
                return torch.LongTensor(inputs), torch.LongTensor([target])[0]
        
        torch_dataset = TorchDataset(dataset)
        return DataLoader(
            torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )
    
    def _create_scheduler(
        self,
        optimizer: optim.Optimizer,
        scheduler_type: Optional[str]
    ) -> Optional[Any]:
        """Create learning rate scheduler."""
        if scheduler_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=50
            )
        else:
            return None
    
    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        optimizer: optim.Optimizer
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': {
                'train_losses': self.history.train_losses,
                'val_losses': self.history.val_losses,
                'train_accuracies': self.history.train_accuracies,
                'val_accuracies': self.history.val_accuracies
            }
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path) -> int:
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Epoch number of checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Optionally restore history
        if 'history' in checkpoint:
            self.history.train_losses = checkpoint['history']['train_losses']
            self.history.val_losses = checkpoint['history']['val_losses']
            self.history.train_accuracies = checkpoint['history']['train_accuracies']
            self.history.val_accuracies = checkpoint['history']['val_accuracies']
        
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['epoch']


def load_music_dataset(
    dataset: str = 'nottingham',
    data_dir: Optional[Path] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    min_length: int = 16,
    max_length: Optional[int] = 128
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """
    Load real music dataset for training.
    
    Args:
        dataset: Dataset name ('nottingham' supported)
        data_dir: Directory to store/load dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        min_length: Minimum melody length
        max_length: Maximum melody length
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if not HAS_DATA_LOADER:
        raise ImportError("data_loader module required for loading music datasets")
    
    if dataset.lower() == 'nottingham':
        logger = logging.getLogger(__name__)
        logger.info("Loading Nottingham dataset...")
        
        loader = NottinghamDataLoader(data_dir=data_dir)
        
        # Try loading processed data first
        processed_path = loader.data_dir / "processed_melodies.json"
        if processed_path.exists():
            logger.info(f"Loading processed data from {processed_path}")
            loader.load_processed_data(processed_path)
            melodies = loader.melodies
        else:
            # Download and parse
            loader.download_dataset()
            melodies, metadata = loader.load_dataset()
            
            # Save processed data for faster loading next time
            loader.save_processed_data(processed_path)
        
        # Show statistics
        stats = loader.get_statistics()
        logger.info(f"Dataset statistics: {stats}")
        
        # Split dataset
        train, val, test = split_dataset(
            melodies,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            min_length=min_length,
            max_length=max_length
        )
        
        return train, val, test
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def create_training_data(
    num_sequences: int = 1000,
    sequence_length: int = 32,
    vocab_size: int = 88,
    train_split: float = 0.8
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Create synthetic training data for melody models.
    
    Args:
        num_sequences: Number of sequences to generate
        sequence_length: Length of each sequence
        vocab_size: Vocabulary size (number of possible notes)
        train_split: Fraction of data for training
        
    Returns:
        Tuple of (train_data, val_data)
    """
    sequences = []
    
    for _ in range(num_sequences):
        # Generate random walk melody
        sequence = [random.randint(0, vocab_size - 1)]
        
        for _ in range(sequence_length - 1):
            # Random walk with bias towards small steps
            step = random.choice([-2, -1, -1, 0, 0, 0, 1, 1, 2])
            next_note = max(0, min(vocab_size - 1, sequence[-1] + step))
            sequence.append(next_note)
        
        sequences.append(sequence)
    
    # Split data
    split_idx = int(len(sequences) * train_split)
    train_data = sequences[:split_idx]
    val_data = sequences[split_idx:]
    
    return train_data, val_data


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    if HAS_TORCH:
        print("Creating example LSTM model and training...")
        
        # Simple LSTM model for demonstration
        class SimpleLSTM(nn.Module):
            def __init__(self, vocab_size=88, hidden_size=128, num_layers=2):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, vocab_size)
            
            def forward(self, x):
                embedded = self.embedding(x)
                lstm_out, _ = self.lstm(embedded)
                output = self.fc(lstm_out[:, -1, :])
                return output
        
        # Create model and trainer
        model = SimpleLSTM()
        trainer = ModelTrainer(model, model_name="example_lstm")
        
        # Create training data
        train_data, val_data = create_training_data(num_sequences=500)
        
        # Train
        history = trainer.train(
            train_data=train_data,
            val_data=val_data,
            epochs=50,
            batch_size=32,
            learning_rate=0.001,
            checkpoint_dir=Path("./checkpoints")
        )
        
        print("\nTraining complete!")
        print(f"Best validation loss: {history.best_val_loss:.4f} at epoch {history.best_epoch}")
    else:
        print("PyTorch not available. Install with: pip install torch")
