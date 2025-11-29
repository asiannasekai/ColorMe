# Model Training and Testing Guide

This guide explains how to train and evaluate the neural network models (LSTM and Transformer) for melody generation in Chromasonic.

## 📚 Overview

The training infrastructure includes:

- **Training Module** (`training.py`): Training with learning curves, validation, and checkpointing
- **Testing Module** (`testing.py`): Comprehensive model evaluation and comparison
- **Training Script** (`train_models.py`): End-to-end training pipeline
- **Data Loader** (`data_loader.py`): Support for real music datasets (Nottingham folk tunes)

## 🎵 Dataset Options

### Synthetic Data (Default)

Quick training with randomly generated melodies:

```bash
python train_models.py --dataset synthetic --num-sequences 1000
```

**Pros:**
- Fast to generate
- Good for testing and debugging
- No download required

**Cons:**
- Random walk patterns (not musical)
- Models won't learn real musical structure

### Nottingham Folk Music Dataset (Recommended)

Train on ~1000 real folk melodies for better results:

```bash
python train_models.py --dataset nottingham --epochs 100
```

**Pros:**
- Real musical patterns
- Better melody structure
- Models learn actual music theory
- Produces more musical outputs

**Cons:**
- Auto-downloads ~5MB dataset on first run
- Slightly longer training time

**Dataset details:**
- ~1000 traditional folk tunes in ABC notation
- Public domain British Isles folk music
- Auto-downloads and parses on first use
- Cached after first load

## 🚀 Quick Start

### Train with Real Music (Recommended)

```bash
cd chromasonic
python train_models.py --dataset nottingham --epochs 100
```

### Train with Synthetic Data

```bash
cd chromasonic
python train_models.py --dataset synthetic --epochs 50 --num-sequences 1000
```

### Train Specific Model

Train only LSTM on Nottingham dataset:
```bash
python train_models.py --model lstm --dataset nottingham --epochs 100
```

Train only Transformer on synthetic data:
```bash
python train_models.py --model transformer --dataset synthetic --epochs 50
```

### Custom Configuration

```bash
python train_models.py \
    --dataset nottingham \
    --model both \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --hidden-size 256 \
    --num-layers 3 \
    --dropout 0.3 \
    --patience 20 \
    --min-melody-length 16 \
    --max-melody-length 128 \
    --output-dir ./my_training_results
```

## 📊 Training Features

### 1. Learning Curves

The training automatically generates comprehensive learning curves showing:

- **Loss curves**: Training vs validation loss over epochs
- **Accuracy curves**: Training vs validation accuracy
- **Learning rate schedule**: LR changes during training
- **Generalization gap**: Overfitting analysis (train loss - val loss)

Example output: `melody_lstm_curves.png`

![Training Curves Example](docs/training_curves_example.png)

### 2. Automatic Checkpointing

Checkpoints are saved automatically:

- **Best model**: Saved when validation loss improves (`*_best.pt`)
- **Periodic checkpoints**: Every 10 epochs (`*_epoch_N.pt`)
- **Final model**: At training completion (`*_final.pt`)

### 3. Early Stopping

Training stops automatically if validation loss doesn't improve for N epochs (default: 15):

```bash
python train_models.py --patience 20  # Stop after 20 epochs without improvement
```

### 4. Learning Rate Scheduling

Three scheduler options available:

- **reduce_on_plateau** (default): Reduce LR when validation plateaus
- **step**: Reduce LR every N epochs
- **cosine**: Cosine annealing schedule

```bash
python train_models.py --lr-scheduler cosine
```

## 🧪 Model Testing

### Comprehensive Test Suite

The testing module evaluates models on multiple metrics:

1. **Prediction Accuracy**: Next-note prediction correctness
2. **Perplexity**: Model confidence (lower is better)
3. **Diversity**: Variety in generated sequences
4. **Coherence**: Melodic smoothness and flow
5. **Musicality**: Overall musical quality

### Running Tests

```python
from chromasonic.melody_generation.testing import ModelTester
from chromasonic.melody_generation.training import create_training_data

# Load your trained model
model = load_trained_model()

# Create test data
_, test_data = create_training_data(num_sequences=200, train_split=0.0)

# Test the model
tester = ModelTester(model, "MyModel")
results = tester.run_comprehensive_test(
    test_data=test_data,
    save_dir=Path("./test_results")
)

print(f"Overall Score: {results['overall_score']:.4f}")
```

### Model Comparison

Compare multiple models side-by-side:

```python
from chromasonic.melody_generation.testing import ModelComparison

comparison = ModelComparison()
results = comparison.compare_models(
    models={
        "LSTM": lstm_model,
        "Transformer": transformer_model,
        "Markov": markov_model
    },
    test_data=test_sequences,
    save_dir=Path("./comparison")
)
```

This generates:
- Detailed comparison table
- Side-by-side bar charts
- Performance radar charts

## 📁 Output Structure

After training, your output directory will contain:

```
training_results/
├── lstm_checkpoints/
│   ├── melody_lstm_best.pt          # Best LSTM model
│   ├── melody_lstm_final.pt         # Final LSTM model
│   ├── melody_lstm_epoch_10.pt      # Periodic checkpoints
│   ├── melody_lstm_history.json     # Training history
│   └── melody_lstm_curves.png       # Learning curves
├── transformer_checkpoints/
│   ├── melody_transformer_best.pt   # Best Transformer model
│   ├── melody_transformer_final.pt  # Final Transformer model
│   ├── melody_transformer_history.json
│   └── melody_transformer_curves.png
└── comparison_results/
    ├── model_comparison.json        # Comparison data
    └── model_comparison.png         # Comparison plots
```

## 🔧 Advanced Usage

### Custom Training Loop

For more control, use the `ModelTrainer` class directly:

```python
from chromasonic.melody_generation.training import ModelTrainer
import torch.nn as nn

# Define your model
class CustomMelodyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture here
    
    def forward(self, x):
        # Your forward pass
        pass

# Create trainer
model = CustomMelodyModel()
trainer = ModelTrainer(model, model_name="custom_model")

# Train
history = trainer.train(
    train_data=train_sequences,
    val_data=val_sequences,
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    checkpoint_dir=Path("./checkpoints")
)

# Plot learning curves
history.plot_curves(save_path="learning_curves.png")
```

### Loading Saved Models

```python
import torch
from pathlib import Path

# Load best checkpoint
checkpoint = torch.load("training_results/lstm_checkpoints/melody_lstm_best.pt")

# Restore model state
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Check which epoch it was from
print(f"Loaded from epoch: {checkpoint['epoch']}")
```

### Analyzing Training History

```python
from chromasonic.melody_generation.training import TrainingHistory

# Load saved history
history = TrainingHistory.load("training_results/lstm_checkpoints/melody_lstm_history.json")

# Access metrics
print(f"Best validation loss: {history.best_val_loss:.4f}")
print(f"Best epoch: {history.best_epoch}")

# Plot again with custom settings
history.plot_curves(save_path="custom_curves.png")
```

## 📈 Interpreting Results

### Learning Curves

**Good Training Signs:**
- Training and validation losses both decrease
- Small gap between train and validation loss
- Smooth curves without erratic jumps

**Warning Signs:**
- Large gap between train and validation (overfitting)
- Validation loss increases while training decreases (overfitting)
- Both losses plateau early (underfitting or too simple model)

### Test Metrics

**Metric Interpretation:**

- **Accuracy (0-1)**: Higher is better
  - >0.7: Excellent
  - 0.5-0.7: Good
  - <0.5: Needs improvement

- **Perplexity**: Lower is better
  - <10: Excellent
  - 10-50: Good
  - >50: Model is uncertain

- **Diversity (0-1)**: Higher is better, but not at expense of coherence
  - >0.7: High variety
  - 0.4-0.7: Balanced
  - <0.4: Too repetitive

- **Coherence (0-1)**: Higher is better
  - >0.7: Smooth melodies
  - 0.5-0.7: Acceptable
  - <0.5: Disjointed

- **Musicality (0-1)**: Higher is better
  - >0.7: Musical structure present
  - 0.5-0.7: Some musical features
  - <0.5: Random-sounding

## 🎯 Best Practices

### Training Tips

1. **Start Small**: Begin with fewer sequences to verify pipeline works
2. **Monitor Overfitting**: Watch the generalization gap plot
3. **Use Early Stopping**: Prevents wasting time on plateaued training
4. **Save Checkpoints**: Keep best model in case final epoch overfits
5. **Try Different Schedulers**: Different tasks benefit from different LR schedules

### Model Selection

- **LSTM**: Faster training, good for shorter sequences, moderate quality
- **Transformer**: Slower training, better for longer sequences, higher quality
- **Markov**: No training needed, fast, baseline comparison

### Hyperparameter Tuning

Start with these ranges:

- **Learning rate**: 0.0001 - 0.01 (try 0.001 first)
- **Batch size**: 16 - 128 (depends on GPU memory)
- **Hidden size**: 64 - 512 (larger = more capacity but slower)
- **Num layers**: 2 - 6 (more layers = more complex patterns)
- **Dropout**: 0.1 - 0.5 (higher for larger models)

## 🐛 Troubleshooting

### Common Issues

**Training loss doesn't decrease:**
- Learning rate too high or too low
- Try: `--learning-rate 0.0001` or `--learning-rate 0.01`

**Overfitting (large train/val gap):**
- Increase dropout: `--dropout 0.3` or `--dropout 0.4`
- Use fewer layers: `--num-layers 2`
- Reduce model size: `--hidden-size 64`
- Get more training data: `--num-sequences 2000`

**Out of memory:**
- Reduce batch size: `--batch-size 16`
- Reduce model size: `--hidden-size 64`
- Reduce number of layers: `--num-layers 2`

**Training too slow:**
- Increase batch size: `--batch-size 64`
- Reduce model size
- Use GPU if available

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Attention Is All You Need (Transformers)](https://arxiv.org/abs/1706.03762)

## 💡 Example Workflow

Complete workflow from training to evaluation:

```bash
# 1. Train both models
python train_models.py \
    --model both \
    --epochs 100 \
    --batch-size 32 \
    --num-sequences 1500 \
    --output-dir ./experiment_1

# 2. Check the learning curves
open ./experiment_1/lstm_checkpoints/melody_lstm_curves.png
open ./experiment_1/transformer_checkpoints/melody_transformer_curves.png

# 3. Review comparison results
open ./experiment_1/comparison_results/model_comparison.png
cat ./experiment_1/comparison_results/model_comparison.json

# 4. Use best model in production
cp ./experiment_1/lstm_checkpoints/melody_lstm_best.pt ./models/production_model.pt
```

---

For more information, see the main Chromasonic documentation.
