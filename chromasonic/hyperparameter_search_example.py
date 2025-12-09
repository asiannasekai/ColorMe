#!/usr/bin/env python3
"""
Hyperparameter Search Example
Demonstrates how to use the hyperparameter optimization module
"""

import sys
from pathlib import Path
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not installed. Install with: pip install torch")
    sys.exit(1)

from chromasonic.melody_generation.training import create_training_data
from chromasonic.melody_generation.hyperparameter_search import (
    HyperparameterSearchSpace,
    HyperparameterOptimizer
)


class SimpleRNN(nn.Module):
    """Simple RNN for melody generation."""
    
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


def main():
    parser = argparse.ArgumentParser(
        description='Run hyperparameter search for melody generation models'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=10,
        help='Number of optimization trials (default: 10)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=15,
        help='Epochs per trial (default: 15)'
    )
    parser.add_argument(
        '--n-sequences',
        type=int,
        default=100,
        help='Number of training sequences to generate (default: 100)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='hyperparameter_search_results',
        help='Output directory for results (default: hyperparameter_search_results)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cpu or cuda)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate optimization plots'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("HYPERPARAMETER OPTIMIZATION FOR MELODY GENERATION")
    print("="*80 + "\n")
    
    # Set up device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU instead")
        device = 'cpu'
    
    print(f"Device: {device}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Epochs per trial: {args.epochs}")
    print(f"Training sequences: {args.n_sequences}\n")
    
    # Generate synthetic training data
    print("Generating training data...")
    train_data, val_data = create_training_data(
        num_sequences=args.n_sequences,
        sequence_length=64,  # Longer sequences to create sliding windows
        vocab_size=88,
        train_split=0.8
    )
    print(f"  Train sequences: {len(train_data)}")
    print(f"  Validation sequences: {len(val_data)}\n")
    
    # Define search space
    search_space = HyperparameterSearchSpace(
        learning_rate_range=(1e-5, 1e-2),
        batch_size_options=[16, 32, 64],
        hidden_size_range=(64, 256),
        num_layers_range=(1, 3),
        dropout_range=(0.0, 0.4),
        embedding_dim_range=(32, 128),
        num_epochs=args.epochs
    )
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        model_class=SimpleRNN,
        search_space=search_space,
        device=device,
        output_dir=Path(args.output_dir),
        verbose=True
    )
    
    # Run optimization
    results = optimizer.optimize(
        train_data=train_data,
        val_data=val_data,
        vocab_size=88,
        sequence_length=16,
        n_trials=args.n_trials,
        study_name="melody_rnn_search"
    )
    
    # Print summary
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    print(f"\nBest Validation Loss: {results['best_value']:.4f}")
    print("\nBest Hyperparameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    print(f"\nResults saved to: {optimizer.output_dir}")
    print("="*80 + "\n")
    
    # Generate plots if requested
    if args.plot:
        print("Generating optimization plots...")
        plot_path = Path(args.output_dir) / 'optimization_history.png'
        optimizer.plot_optimization_history(save_path=plot_path)
    
    print("✓ Hyperparameter search completed!")


if __name__ == '__main__':
    main()
