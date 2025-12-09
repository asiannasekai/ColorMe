# Hyperparameter Search Guide

## Overview

This guide explains how to use the hyperparameter optimization module to automatically find the best hyperparameters for your melody generation models.

## What is Hyperparameter Tuning?

Hyperparameter tuning is the process of automatically searching for the best combination of model settings (hyperparameters) to optimize performance. This is different from learning parameters, which are learned during training.

Common hyperparameters we optimize:
- **Learning Rate**: Controls how much weights change in each step
- **Batch Size**: Number of samples processed before updating weights
- **Hidden Size**: Number of units in hidden layers
- **Number of Layers**: Depth of the neural network
- **Dropout**: Regularization technique to prevent overfitting
- **Embedding Dimension**: Size of the embedding layer

## Quick Start

### 1. Install Optuna

```bash
pip install optuna
```

Or update your environment:

```bash
pip install -r requirements.txt
```

### 2. Run Basic Hyperparameter Search

```bash
cd chromasonic
python hyperparameter_search_example.py --n-trials 10 --epochs 15
```

### 3. View Results

Results are saved in the `hyperparameter_search_results` directory:
- `best_hyperparameters.json` - The best parameters found
- `optimization_results.json` - Detailed results for all trials
- `optimization_report.txt` - Human-readable summary

## Advanced Usage

### Custom Search Space

Modify the search space in your script:

```python
from chromasonic.melody_generation.hyperparameter_search import (
    HyperparameterSearchSpace,
    HyperparameterOptimizer
)

# Define your custom search space
search_space = HyperparameterSearchSpace(
    learning_rate_range=(1e-6, 1e-1),      # Wider range
    batch_size_options=[8, 16, 32, 64, 128],  # More options
    hidden_size_range=(32, 1024),          # Larger networks
    num_layers_range=(1, 5),
    dropout_range=(0.0, 0.6),
    embedding_dim_range=(16, 512),
    num_epochs=50                          # More epochs per trial
)

optimizer = HyperparameterOptimizer(
    model_class=YourModelClass,
    search_space=search_space,
    device='cuda',
    output_dir=Path('my_search_results')
)

results = optimizer.optimize(
    train_data=train_data,
    val_data=val_data,
    vocab_size=88,
    n_trials=50,  # Run more trials
    timeout=3600  # 1 hour timeout
)
```

### Command Line Arguments

```bash
python hyperparameter_search_example.py \
    --n-trials 20              # Number of optimization trials
    --epochs 20                # Epochs per trial
    --n-sequences 500          # Training sequences
    --output-dir results       # Output directory
    --device cuda              # Use GPU
    --plot                     # Generate plots
```

### Using Different Model Architectures

```python
from chromasonic.melody_generation.models import TransformerMelodyModel

# Use your own model class
optimizer = HyperparameterOptimizer(
    model_class=TransformerMelodyModel,  # Your custom model
    search_space=search_space,
    device='cuda'
)
```

## Understanding the Results

### optimization_results.json

```json
{
  "timestamp": "2024-01-15T10:30:45.123456",
  "total_time": 3600.5,
  "best_params": {
    "learning_rate": 0.0015,
    "batch_size": 32,
    "hidden_size": 192,
    "num_layers": 2,
    "dropout": 0.2,
    "embedding_dim": 64
  },
  "best_value": 2.1234,
  "n_trials": 20,
  "trial_results": [...]
}
```

### optimization_report.txt

Contains a ranked list of the top 10 configurations:

```
RANK 1: Trial 5
  Val Loss: 2.1234
  Train Loss: 2.0456
  Best Epoch: 12
  Training Time: 156.23s
  Parameters: 127,488
  Hyperparameters:
    learning_rate: 0.0015
    batch_size: 32
    hidden_size: 192
    ...
```

## Key Concepts

### Objective Function

The optimizer minimizes the **validation loss** as the objective. Lower validation loss generally indicates better model performance.

### Early Stopping and Pruning

- **Pruning**: The optimizer stops trials that are performing poorly early, saving computation time
- **Early Stopping**: Individual trials stop if validation loss doesn't improve for 5 epochs

### Bayesian Optimization

The optimizer uses **Tree-structured Parzen Estimator (TPE)**, a Bayesian optimization algorithm that:
1. Learns from previous trials
2. Intelligently suggests promising hyperparameter combinations
3. Gradually focuses on the best regions of the search space

## Tips for Better Results

1. **Increase Trials**: More trials = better coverage of the search space
   ```bash
   --n-trials 50  # Better results than 10
   ```

2. **Longer Training**: More epochs per trial = better convergence
   ```bash
   --epochs 30  # Better than 10
   ```

3. **More Data**: More sequences = more robust optimization
   ```bash
   --n-sequences 1000  # Better than 100
   ```

4. **Focus Search Space**: 
   - If you find good learning rates around 0.001, narrow the range
   - If hidden_size=256 is always best, increase that range
   - Reduce options for parameters that don't matter

5. **Use GPU**: Significantly speeds up optimization
   ```bash
   --device cuda
   ```

## Complete Example

```python
#!/usr/bin/env python3
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Your imports
from chromasonic.melody_generation.training import create_training_data
from chromasonic.melody_generation.hyperparameter_search import (
    HyperparameterSearchSpace,
    HyperparameterOptimizer
)

# Your model class
class MyMelodyModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        super().__init__()
        # Your model implementation
        pass
    
    def forward(self, x):
        # Your forward pass
        pass

# Main script
def main():
    # Create training data
    train_data, val_data = create_training_data(
        num_sequences=500,
        sequence_length=16,
        vocab_size=88,
        train_split=0.8
    )
    
    # Define search space
    search_space = HyperparameterSearchSpace(
        learning_rate_range=(1e-5, 1e-2),
        batch_size_options=[16, 32, 64, 128],
        hidden_size_range=(64, 512),
        num_layers_range=(1, 4),
        dropout_range=(0.0, 0.5),
        embedding_dim_range=(32, 256),
        num_epochs=20
    )
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        model_class=MyMelodyModel,
        search_space=search_space,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        output_dir=Path('hyperparameter_results')
    )
    
    # Run optimization
    results = optimizer.optimize(
        train_data=train_data,
        val_data=val_data,
        vocab_size=88,
        n_trials=50,
        study_name="my_melody_search"
    )
    
    # Generate plots
    optimizer.plot_optimization_history(
        save_path=Path('hyperparameter_results/plots.png')
    )
    
    # Print best hyperparameters
    print("Best Hyperparameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    
    # Use best parameters to train final model
    model = MyMelodyModel(
        vocab_size=88,
        **results['best_params']
    )
    
    # Continue with training...

if __name__ == '__main__':
    main()
```

## Troubleshooting

### Out of Memory (OOM)

If you get CUDA out of memory errors:
- Reduce batch sizes
- Use CPU: `--device cpu`
- Reduce number of epochs per trial
- Use smaller models

### Slow Optimization

If it's taking too long:
- Reduce number of trials: `--n-trials 10`
- Reduce epochs per trial: `--epochs 10`
- Use early stopping (default 5 epochs without improvement)

### Poor Results

If results aren't improving:
- Try a wider search space
- Increase number of trials to 50+
- Check if your model has bugs
- Ensure training data is appropriate

## Next Steps

1. **Use Best Parameters**: Once you find the best hyperparameters, train your final model with them
2. **Ensemble**: Train multiple models with slightly different top hyperparameters
3. **Fine-tune**: Do another search with narrower ranges around the best values
4. **Test**: Always evaluate on a held-out test set

## Further Reading

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Hyperparameter Optimization in Deep Learning](https://arxiv.org/abs/1810.03779)
- [Bayesian Optimization](https://arxiv.org/abs/1807.02811)
