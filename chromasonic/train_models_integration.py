#!/usr/bin/env python3
"""
Integration Example: Using Hyperparameter Search with Existing Training Pipeline

This shows how to integrate the hyperparameter optimization with your existing
train_models.py workflow.
"""

import sys
from pathlib import Path
import json
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not installed")
    sys.exit(1)

from chromasonic.melody_generation.training import (
    ModelTrainer,
    create_training_data
)
from chromasonic.melody_generation.hyperparameter_search import (
    HyperparameterSearchSpace,
    HyperparameterOptimizer
)


class SimpleRNN(nn.Module):
    """Simple RNN model (same as in train_models.py)"""
    
    def __init__(
        self,
        vocab_size: int = 88,
        embedding_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        last_output = rnn_out[:, -1, :]
        dropped = self.dropout(last_output)
        output = self.fc(dropped)
        return output


class TransformerMelodyModel(nn.Module):
    """Transformer model for melody generation"""
    
    def __init__(
        self,
        vocab_size: int = 88,
        embedding_dim: int = 64,
        hidden_size: int = 256,
        num_layers: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.pos_encoder = nn.Parameter(
            torch.randn(1, 100, embedding_dim)  # Max seq length 100
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.fc = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x):
        seq_len = x.size(1)
        embedded = self.embedding(x) * (self.pos_encoder.size(-1) ** 0.5)
        embedded = embedded + self.pos_encoder[:, :seq_len, :]
        
        transformer_out = self.transformer(embedded)
        last_output = transformer_out[:, -1, :]
        output = self.fc(last_output)
        
        return output


def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter search integrated with training pipeline'
    )
    parser.add_argument(
        '--mode',
        choices=['search', 'train_best'],
        default='search',
        help='search: run hyperparameter search, train_best: train with best params'
    )
    parser.add_argument(
        '--model',
        choices=['rnn', 'transformer'],
        default='rnn',
        help='Model architecture to optimize'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=10,
        help='Number of optimization trials'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='Epochs per trial'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    parser.add_argument(
        '--best-params-file',
        type=str,
        default='hyperparameter_search_results/best_hyperparameters.json',
        help='File containing best hyperparameters'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'search':
        run_hyperparameter_search(args)
    else:
        train_with_best_params(args)


def run_hyperparameter_search(args):
    """Run hyperparameter search"""
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH FOR MELODY GENERATION")
    print("="*80 + "\n")
    
    # Select model class
    if args.model == 'rnn':
        model_class = SimpleRNN
        print("Model: Simple RNN")
    else:
        model_class = TransformerMelodyModel
        print("Model: Transformer")
    
    # Generate training data
    print("\nGenerating training data...")
    train_data, val_data = create_training_data(
        num_sequences=100,
        sequence_length=16,
        vocab_size=88,
        train_split=0.8
    )
    print(f"  Train sequences: {len(train_data)}")
    print(f"  Validation sequences: {len(val_data)}")
    
    # Configure search space
    if args.model == 'rnn':
        search_space = HyperparameterSearchSpace(
            learning_rate_range=(1e-5, 1e-2),
            batch_size_options=[16, 32, 64],
            hidden_size_range=(64, 256),
            num_layers_range=(1, 3),
            dropout_range=(0.0, 0.4),
            embedding_dim_range=(32, 128),
            num_epochs=args.epochs
        )
    else:  # Transformer
        search_space = HyperparameterSearchSpace(
            learning_rate_range=(1e-5, 1e-3),
            batch_size_options=[16, 32],
            hidden_size_range=(256, 512),
            num_layers_range=(2, 6),
            dropout_range=(0.1, 0.4),
            embedding_dim_range=(64, 256),
            num_epochs=args.epochs
        )
    
    # Run optimization
    optimizer = HyperparameterOptimizer(
        model_class=model_class,
        search_space=search_space,
        device=args.device,
        output_dir=Path('hyperparameter_search_results'),
        verbose=True
    )
    
    results = optimizer.optimize(
        train_data=train_data,
        val_data=val_data,
        vocab_size=88,
        n_trials=args.n_trials,
        study_name=f"melody_{args.model}_search"
    )
    
    print("\n" + "="*80)
    print("SEARCH RESULTS")
    print("="*80)
    print(f"Best Validation Loss: {results['best_value']:.4f}")
    print("\nBest Hyperparameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    
    print(f"\nResults saved to: hyperparameter_search_results/")
    print("  - best_hyperparameters.json")
    print("  - optimization_results.json")
    print("  - optimization_report.txt")
    
    # Generate plots
    try:
        print("\nGenerating plots...")
        plot_path = Path('hyperparameter_search_results/optimization_history.png')
        optimizer.plot_optimization_history(save_path=plot_path)
        print(f"  Saved to: {plot_path}")
    except Exception as e:
        print(f"  Could not generate plots: {e}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Review the best hyperparameters found")
    print("2. Train a final model with: python train_models_integration.py --mode train_best")
    print("3. Evaluate on test set")
    print("="*80 + "\n")


def train_with_best_params(args):
    """Train a model using the best hyperparameters from search"""
    print("\n" + "="*80)
    print("TRAINING WITH BEST HYPERPARAMETERS")
    print("="*80 + "\n")
    
    # Load best hyperparameters
    params_file = Path(args.best_params_file)
    if not params_file.exists():
        print(f"Error: Best parameters file not found at {params_file}")
        print("Run with --mode search first")
        return
    
    with open(params_file, 'r') as f:
        best_params = json.load(f)
    
    print("Best Hyperparameters Found:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Select model class
    if args.model == 'rnn':
        model_class = SimpleRNN
        print("\nModel: Simple RNN")
    else:
        model_class = TransformerMelodyModel
        print("\nModel: Transformer")
    
    # Generate training data (you could load real data here)
    print("\nGenerating training data...")
    train_data, val_data = create_training_data(
        num_sequences=500,
        sequence_length=16,
        vocab_size=88,
        train_split=0.8
    )
    print(f"  Train sequences: {len(train_data)}")
    print(f"  Validation sequences: {len(val_data)}")
    
    # Create model with best parameters
    print("\nCreating model...")
    model = model_class(
        vocab_size=88,
        embedding_dim=best_params['embedding_dim'],
        hidden_size=best_params['hidden_size'],
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nTraining model...")
    trainer = ModelTrainer(model, model_name=f"final_{args.model}_model", device=args.device)
    
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=100,
        batch_size=best_params['batch_size'],
        learning_rate=best_params['learning_rate'],
        sequence_length=16,
        early_stopping_patience=10,
        checkpoint_dir=Path('final_model_checkpoints'),
        lr_scheduler='cosine'
    )
    
    # Save results
    print("\nTraining complete!")
    print(f"Final validation loss: {history.val_losses[-1]:.4f}")
    print(f"Best validation loss: {history.best_val_loss:.4f} (epoch {history.best_epoch})")
    
    # Plot learning curves
    print("\nGenerating learning curves...")
    history.plot_learning_curves(save_path=Path('final_model_results/learning_curves.png'))
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("Results saved to: final_model_results/")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
