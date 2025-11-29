#!/usr/bin/env python3
"""
Example Training Script for Chromasonic Melody Models

This script demonstrates how to train LSTM and Transformer models
with comprehensive evaluation and visualization.
"""

import sys
from pathlib import Path
import logging
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from chromasonic.melody_generation.training import (
    ModelTrainer,
    TrainingHistory,
    create_training_data
)
from chromasonic.melody_generation.testing import (
    ModelTester,
    ModelComparison
)

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not installed. Install with: pip install torch")
    sys.exit(1)


class SimpleRNN(nn.Module):
    """Vanilla RNN model for melody generation (baseline comparison)."""
    
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
        # Using vanilla RNN instead of LSTM - no memory cells
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
        """Forward pass."""
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        
        # Use only the last output
        last_output = rnn_out[:, -1, :]
        dropped = self.dropout(last_output)
        output = self.fc(dropped)
        
        return output
    
    def generate(self, seed: list, length: int = 16) -> list:
        """Generate a sequence from seed."""
        self.eval()
        
        generated = seed.copy()
        
        with torch.no_grad():
            for _ in range(length - len(seed)):
                # Prepare input
                input_seq = torch.LongTensor([generated[-16:]])  # Use last 16 notes
                
                # Predict
                output = self.forward(input_seq)
                probs = torch.softmax(output[0], dim=0)
                
                # Sample from distribution
                next_note = torch.multinomial(probs, 1).item()
                generated.append(next_note)
        
        return generated


class MelodyLSTM(nn.Module):
    """LSTM model for melody generation."""
    
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
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        # Use only the last output
        last_output = lstm_out[:, -1, :]
        dropped = self.dropout(last_output)
        output = self.fc(dropped)
        
        return output
    
    def generate(self, seed: list, length: int = 16) -> list:
        """Generate a sequence from seed."""
        self.eval()
        
        generated = seed.copy()
        
        with torch.no_grad():
            for _ in range(length - len(seed)):
                # Prepare input
                input_seq = torch.LongTensor([generated[-16:]])  # Use last 16 notes
                
                # Predict
                output = self.forward(input_seq)
                probs = torch.softmax(output[0], dim=0)
                
                # Sample from distribution
                next_note = torch.multinomial(probs, 1).item()
                generated.append(next_note)
        
        return generated


class MelodyTransformer(nn.Module):
    """Transformer model for melody generation."""
    
    def __init__(
        self,
        vocab_size: int = 88,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        """Forward pass."""
        # x shape: (batch_size, sequence_length)
        seq_len = x.size(1)
        
        embedded = self.embedding(x) * (self.d_model ** 0.5)
        embedded = embedded + self.pos_encoder[:, :seq_len, :]
        
        transformer_out = self.transformer(embedded)
        
        # Use only the last output
        last_output = transformer_out[:, -1, :]
        output = self.fc(last_output)
        
        return output
    
    def generate(self, seed: list, length: int = 16) -> list:
        """Generate a sequence from seed."""
        self.eval()
        
        generated = seed.copy()
        
        with torch.no_grad():
            for _ in range(length - len(seed)):
                # Prepare input
                input_seq = torch.LongTensor([generated[-16:]])
                
                # Predict
                output = self.forward(input_seq)
                probs = torch.softmax(output[0], dim=0)
                
                # Sample from distribution
                next_note = torch.multinomial(probs, 1).item()
                generated.append(next_note)
        
        return generated


def train_rnn_model(args, train_data=None, val_data=None):
    """Train vanilla RNN model for baseline comparison."""
    print("\n" + "="*80)
    print("Training Simple RNN Model (Baseline)")
    print("="*80 + "\n")
    
    # Create model
    model = SimpleRNN(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create training data if not provided
    if train_data is None or val_data is None:
        print(f"\nGenerating {args.num_sequences} training sequences...")
        train_data, val_data = create_training_data(
            num_sequences=args.num_sequences,
            sequence_length=args.sequence_length,
            vocab_size=args.vocab_size,
            train_split=0.8
        )
    
    print(f"Train sequences: {len(train_data)}")
    print(f"Validation sequences: {len(val_data)}")
    
    # Create trainer
    trainer = ModelTrainer(model, model_name="melody_rnn")
    
    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sequence_length=16,
        early_stopping_patience=args.patience,
        checkpoint_dir=Path(args.output_dir) / "rnn_checkpoints",
        lr_scheduler=args.lr_scheduler
    )
    
    # Test model
    print("\nTesting RNN model...")
    tester = ModelTester(model, model_name="Simple RNN")
    
    test_results = tester.evaluate(val_data[:100], sequence_length=16)
    
    print("\n" + "="*80)
    print("RNN MODEL RESULTS")
    print("="*80)
    print(f"Final Training Loss: {history.train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {history.val_losses[-1]:.4f}")
    print(f"Test Accuracy: {test_results['accuracy']:.2%}")
    print(f"Test Perplexity: {test_results['perplexity']:.2f}")
    
    # Save results
    output_dir = Path(args.output_dir) / "rnn_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot learning curves
    history.plot_learning_curves(save_path=output_dir / "learning_curves.png")
    
    # Generate samples
    seed = list(range(60, 68))  # C major scale
    samples = tester.generate_samples(
        num_samples=5,
        seed_sequence=seed,
        length=32,
        temperature=1.0
    )
    
    print(f"\nResults saved to: {output_dir}")
    
    return model, history, test_results


def train_lstm_model(args, train_data=None, val_data=None):
    """Train LSTM model."""
    print("\n" + "="*80)
    print("Training LSTM Model")
    print("="*80 + "\n")
    
    # Create model
    model = MelodyLSTM(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create training data if not provided
    if train_data is None or val_data is None:
        print(f"\nGenerating {args.num_sequences} training sequences...")
        train_data, val_data = create_training_data(
            num_sequences=args.num_sequences,
            sequence_length=args.sequence_length,
            vocab_size=args.vocab_size,
            train_split=0.8
        )
    
    print(f"Train sequences: {len(train_data)}")
    print(f"Validation sequences: {len(val_data)}")
    
    # Create trainer
    trainer = ModelTrainer(model, model_name="melody_lstm")
    
    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sequence_length=16,
        early_stopping_patience=args.patience,
        checkpoint_dir=Path(args.output_dir) / "lstm_checkpoints",
        lr_scheduler=args.lr_scheduler
    )
    
    return model, history


def train_transformer_model(args, train_data=None, val_data=None):
    """Train Transformer model."""
    print("\n" + "="*80)
    print("Training Transformer Model")
    print("="*80 + "\n")
    
    # Create model
    model = MelodyTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    print(f"Model architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create training data if not provided
    if train_data is None or val_data is None:
        print(f"\nGenerating {args.num_sequences} training sequences...")
        train_data, val_data = create_training_data(
            num_sequences=args.num_sequences,
            sequence_length=args.sequence_length,
            vocab_size=args.vocab_size,
            train_split=0.8
        )
    
    print(f"Train sequences: {len(train_data)}")
    print(f"Validation sequences: {len(val_data)}")
    
    # Create trainer
    trainer = ModelTrainer(model, model_name="melody_transformer")
    
    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sequence_length=16,
        early_stopping_patience=args.patience,
        checkpoint_dir=Path(args.output_dir) / "transformer_checkpoints",
        lr_scheduler=args.lr_scheduler
    )
    
    return model, history


def test_and_compare_models(lstm_model, transformer_model, args):
    """Test and compare trained models."""
    print("\n" + "="*80)
    print("Testing and Comparing Models")
    print("="*80 + "\n")
    
    # Create test data
    _, test_data = create_training_data(
        num_sequences=args.num_test_sequences,
        sequence_length=args.sequence_length,
        vocab_size=args.vocab_size,
        train_split=0.0
    )
    
    print(f"Test sequences: {len(test_data)}")
    
    # Compare models
    comparison = ModelComparison()
    results = comparison.compare_models(
        models={
            "LSTM": lstm_model,
            "Transformer": transformer_model
        },
        test_data=test_data,
        save_dir=Path(args.output_dir) / "comparison_results"
    )
    
    return results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate melody generation models"
    )
    
    # Data parameters
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'nottingham'],
                       help='Dataset to use for training')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Directory for dataset storage (default: ./data/nottingham)')
    parser.add_argument('--num-sequences', type=int, default=1000,
                       help='Number of training sequences to generate (synthetic only)')
    parser.add_argument('--num-test-sequences', type=int, default=200,
                       help='Number of test sequences (synthetic only)')
    parser.add_argument('--sequence-length', type=int, default=32,
                       help='Length of each sequence (synthetic only)')
    parser.add_argument('--min-melody-length', type=int, default=16,
                       help='Minimum melody length for real datasets')
    parser.add_argument('--max-melody-length', type=int, default=128,
                       help='Maximum melody length for real datasets')
    parser.add_argument('--vocab-size', type=int, default=88,
                       help='Vocabulary size (number of possible notes)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--lr-scheduler', type=str, default='reduce_on_plateau',
                       choices=['reduce_on_plateau', 'step', 'cosine', 'none'],
                       help='Learning rate scheduler')
    
    # Model parameters (LSTM)
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM/Transformer layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Model parameters (Transformer)
    parser.add_argument('--d-model', type=int, default=128,
                       help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./training_results',
                       help='Output directory for checkpoints and results')
    
    # Model selection
    parser.add_argument('--model', type=str, default='all',
                       choices=['rnn', 'lstm', 'transformer', 'all'],
                       help='Which model(s) to train')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Chromasonic Melody Model Training")
    print("="*80)
    print(f"\nConfiguration:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    
    # Load or generate training data
    print("\n" + "="*80)
    if args.dataset == 'nottingham':
        print("Loading Nottingham Dataset")
        print("="*80)
        from src.chromasonic.melody_generation.training import load_music_dataset
        
        data_dir = Path(args.data_dir) if args.data_dir else None
        train_data, val_data, test_data = load_music_dataset(
            dataset='nottingham',
            data_dir=data_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            min_length=args.min_melody_length,
            max_length=args.max_melody_length
        )
        print(f"\n✓ Loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test melodies")
        
    else:  # synthetic
        print("Generating Synthetic Data")
        print("="*80)
        train_data, val_data = create_training_data(
            num_sequences=args.num_sequences,
            sequence_length=args.sequence_length,
            vocab_size=args.vocab_size,
            train_split=0.8
        )
        
        # Generate test data
        test_data, _ = create_training_data(
            num_sequences=args.num_test_sequences,
            sequence_length=args.sequence_length,
            vocab_size=args.vocab_size,
            train_split=1.0
        )
        print(f"\n✓ Generated: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test sequences")
    
    # Train models
    rnn_model = None
    lstm_model = None
    transformer_model = None
    
    if args.model in ['rnn', 'all']:
        rnn_model, rnn_history, rnn_results = train_rnn_model(args, train_data, val_data)
    
    if args.model in ['lstm', 'all']:
        lstm_model, lstm_history = train_lstm_model(args, train_data, val_data)
    
    if args.model in ['transformer', 'all']:
        transformer_model, transformer_history = train_transformer_model(args, train_data, val_data)
    
    # Compare models if multiple were trained
    if args.model == 'all' and lstm_model and transformer_model:
        test_and_compare_models(lstm_model, transformer_model, args)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - Model checkpoints (*.pt)")
    print("  - Training history (*.json)")
    print("  - Learning curves (*.png)")
    print("  - Test results and comparisons")
    print("\n")


if __name__ == "__main__":
    main()
