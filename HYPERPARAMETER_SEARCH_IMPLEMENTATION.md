# Hyperparameter Search Implementation Summary

## What Was Added

A complete hyperparameter optimization system using **Optuna** (industry-standard Bayesian optimization framework) has been added to the ColorMe/Chromasonic project.

## Files Created

### 1. `/chromasonic/src/chromasonic/melody_generation/hyperparameter_search.py`
Core hyperparameter optimization module containing:

- **`HyperparameterSearchSpace`**: Defines the search space for hyperparameters
  - Learning rate range (log scale)
  - Batch size options
  - Hidden layer sizes
  - Number of layers
  - Dropout rates
  - Embedding dimensions
  - Epochs per trial

- **`HyperparameterOptimizer`**: Main optimizer class
  - Objective function for trial evaluation
  - Automatic trial management
  - Result tracking and reporting
  - JSON export of results
  - Summary report generation
  - Visualization support

### 2. `/chromasonic/hyperparameter_search_example.py`
Complete runnable example demonstrating:
- How to set up the optimizer
- How to configure the search space
- How to run optimization trials
- How to visualize results
- Command-line interface with arguments

### 3. `/chromasonic/HYPERPARAMETER_SEARCH.md`
Comprehensive guide covering:
- Installation instructions
- Quick start examples
- Advanced usage patterns
- Result interpretation
- Troubleshooting
- Complete working example

### 4. Updated `/chromasonic/requirements.txt`
Added `optuna>=3.0.0` for hyperparameter optimization

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│  Define Search Space (learning rate, batch size, etc)   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Trial 1: Suggest hyperparameters                       │
│  - Train model with those params                        │
│  - Evaluate on validation set                           │
│  - Return validation loss                               │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │ Learn from Trial 1      │
        │ (Bayesian Optimizer)    │
        └────────────┬────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Trial 2: Suggest better hyperparameters               │
│  - Train model with new params                          │
│  - Evaluate on validation set                           │
│  - Return validation loss                               │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │ Learn from Trial 1 & 2  │
        │ (Bayesian Optimizer)    │
        └────────────┬────────────┘
                     │
                     ▼
            ... (repeat N times) ...
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Return Best Hyperparameters Found                      │
│  - Best validation loss value                           │
│  - Corresponding hyperparameter config                  │
│  - Full optimization history                            │
└─────────────────────────────────────────────────────────┘
```

## Key Features

✅ **Bayesian Optimization**: Uses TPE (Tree-structured Parzen Estimator) sampler for intelligent search  
✅ **Automatic Early Stopping**: Stops unpromising trials early to save computation  
✅ **Comprehensive Logging**: Detailed progress information and results  
✅ **Result Persistence**: Saves best parameters and optimization history as JSON  
✅ **Visualization**: Plots optimization history and parameter impact  
✅ **Flexible Search Space**: Easy to customize what parameters to optimize  
✅ **Multi-Trial Tracking**: Detailed metrics for each trial  
✅ **Device Support**: Works with both CPU and GPU (CUDA)  

## Quick Start

### 1. Install Optuna
```bash
pip install optuna
# Or update all dependencies:
pip install -r requirements.txt
```

### 2. Run Hyperparameter Search
```bash
cd chromasonic
python hyperparameter_search_example.py --n-trials 10 --epochs 15
```

### 3. Find Results
```bash
cat hyperparameter_search_results/best_hyperparameters.json
cat hyperparameter_search_results/optimization_report.txt
```

## Example Usage

```python
from chromasonic.melody_generation.hyperparameter_search import (
    HyperparameterSearchSpace,
    HyperparameterOptimizer
)

# Define search space
search_space = HyperparameterSearchSpace(
    learning_rate_range=(1e-5, 1e-2),
    batch_size_options=[16, 32, 64],
    hidden_size_range=(64, 512),
    num_layers_range=(1, 4),
    dropout_range=(0.0, 0.5),
    embedding_dim_range=(32, 256),
    num_epochs=20
)

# Create optimizer
optimizer = HyperparameterOptimizer(
    model_class=SimpleRNN,
    search_space=search_space,
    device='cuda'
)

# Run optimization
results = optimizer.optimize(
    train_data=train_data,
    val_data=val_data,
    vocab_size=88,
    n_trials=20  # Run 20 trials
)

# Access best parameters
print(results['best_params'])
# Output: {'learning_rate': 0.0015, 'batch_size': 32, ...}
```

## Output Files

After running optimization, you'll get:

1. **`best_hyperparameters.json`** - The single best configuration found
2. **`optimization_results.json`** - Complete results for all trials
3. **`optimization_report.txt`** - Human-readable summary with top 10 trials
4. **`optimization_history.png`** - Visualization of optimization process

## What Can Be Optimized

- **learning_rate**: How fast the model learns (log scale search for efficiency)
- **batch_size**: Number of samples per gradient update
- **hidden_size**: Number of units in hidden layers
- **num_layers**: Depth of the neural network
- **dropout**: Regularization to prevent overfitting
- **embedding_dim**: Size of the embedding layer

All these can be easily customized in `HyperparameterSearchSpace`.

## Performance Improvements

Based on typical neural network tuning:
- Without tuning: validation loss might be ~2.5
- With tuning (10 trials): validation loss might be ~2.2
- With tuning (50 trials): validation loss might be ~2.0
- With tuning (100 trials): validation loss might be ~1.8-1.9

The exact improvement depends on your model and data.

## Next Steps

1. **Install Optuna**: `pip install optuna`
2. **Read the Guide**: Review `HYPERPARAMETER_SEARCH.md` for detailed instructions
3. **Run Example**: Execute `hyperparameter_search_example.py`
4. **Customize**: Modify the search space for your specific needs
5. **Integrate**: Use found hyperparameters in your main training pipeline

## Technical Details

- **Sampler**: Tree-structured Parzen Estimator (TPE) - learns to sample promising regions
- **Pruner**: Median Pruner - stops unpromising trials early
- **Objective**: Minimizes validation loss
- **Framework**: Optuna 3.0+ (latest version recommended)

## Troubleshooting

- **"Optuna not installed"**: Run `pip install optuna`
- **Out of Memory**: Reduce batch sizes or use CPU
- **Too Slow**: Reduce number of trials or epochs per trial
- **Poor Results**: Try a wider search space or more trials

## Comparison with Other Approaches

| Feature | Grid Search | Random Search | **Bayesian (Optuna)** |
|---------|-------------|---------------|----------------------|
| Efficiency | ❌ Slow | ⚠️ Medium | ✅ Fast |
| Learning | ❌ No | ❌ No | ✅ Yes |
| Early Stopping | ❌ No | ❌ No | ✅ Yes |
| Scalability | ❌ Poor | ⚠️ Medium | ✅ Excellent |
| Ease of Use | ✅ Simple | ✅ Simple | ✅ Simple |

## References

- [Optuna Official Documentation](https://optuna.readthedocs.io/)
- [Practical Bayesian Optimization of Machine Learning Algorithms](https://arxiv.org/abs/1206.2944)
- [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06393)
