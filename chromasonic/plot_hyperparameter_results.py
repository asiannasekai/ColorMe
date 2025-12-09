#!/usr/bin/env python3
"""
Visualize hyperparameter search results with multiple graphs
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Load results
results_path = Path('hyperparameter_search_results/optimization_results.json')
with open(results_path, 'r') as f:
    results = json.load(f)

# Extract data
trials = results['trial_results']
trial_nums = [t['trial'] for t in trials]
final_losses = [t['final_val_loss'] for t in trials]
best_losses = [t['best_val_loss'] for t in trials]
training_times = [t['training_time'] for t in trials]
num_params = [t['num_parameters'] for t in trials]

# Extract hyperparameters
learning_rates = [t['params']['learning_rate'] for t in trials]
batch_sizes = [t['params']['batch_size'] for t in trials]
hidden_sizes = [t['params']['hidden_size'] for t in trials]
num_layers = [t['params']['num_layers'] for t in trials]
dropouts = [t['params']['dropout'] for t in trials]
embedding_dims = [t['params']['embedding_dim'] for t in trials]

# Create figure with subplots
fig = plt.figure(figsize=(18, 14))

# 1. Validation Loss over Trials
ax1 = plt.subplot(3, 3, 1)
ax1.plot(trial_nums, final_losses, 'o-', linewidth=2, markersize=8, label='Final Val Loss', color='steelblue')
ax1.axhline(y=results['best_value'], color='r', linestyle='--', linewidth=2, label=f"Best Loss: {results['best_value']:.4f}")
ax1.set_xlabel('Trial Number', fontsize=11, fontweight='bold')
ax1.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
ax1.set_title('Validation Loss Progression', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Learning Rate vs Validation Loss
ax2 = plt.subplot(3, 3, 2)
scatter = ax2.scatter(learning_rates, final_losses, c=trial_nums, cmap='viridis', s=100, alpha=0.6, edgecolors='black', linewidth=1)
ax2.set_xlabel('Learning Rate (log scale)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
ax2.set_xscale('log')
ax2.set_title('Learning Rate Impact', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax2, label='Trial')
ax2.grid(True, alpha=0.3)

# 3. Batch Size vs Validation Loss
ax3 = plt.subplot(3, 3, 3)
batch_size_unique = sorted(set(batch_sizes))
batch_losses = [np.mean([final_losses[i] for i in range(len(trials)) if batch_sizes[i] == bs]) for bs in batch_size_unique]
ax3.bar(range(len(batch_size_unique)), batch_losses, color='coral', edgecolor='black', linewidth=1.5, alpha=0.7)
ax3.set_xticks(range(len(batch_size_unique)))
ax3.set_xticklabels(batch_size_unique)
ax3.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
ax3.set_ylabel('Average Validation Loss', fontsize=11, fontweight='bold')
ax3.set_title('Batch Size Impact', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Hidden Size vs Validation Loss
ax4 = plt.subplot(3, 3, 4)
ax4.scatter(hidden_sizes, final_losses, s=100, alpha=0.6, edgecolors='black', linewidth=1, color='lightgreen')
ax4.set_xlabel('Hidden Size', fontsize=11, fontweight='bold')
ax4.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
ax4.set_title('Hidden Size Impact', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. Number of Layers vs Validation Loss
ax5 = plt.subplot(3, 3, 5)
layer_unique = sorted(set(num_layers))
layer_losses = [np.mean([final_losses[i] for i in range(len(trials)) if num_layers[i] == nl]) for nl in layer_unique]
colors_layers = ['#FF6B6B', '#4ECDC4', '#45B7D1']
ax5.bar(range(len(layer_unique)), layer_losses, color=colors_layers[:len(layer_unique)], edgecolor='black', linewidth=1.5, alpha=0.7)
ax5.set_xticks(range(len(layer_unique)))
ax5.set_xticklabels(layer_unique)
ax5.set_xlabel('Number of Layers', fontsize=11, fontweight='bold')
ax5.set_ylabel('Average Validation Loss', fontsize=11, fontweight='bold')
ax5.set_title('Number of Layers Impact', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# 6. Dropout vs Validation Loss
ax6 = plt.subplot(3, 3, 6)
ax6.scatter(dropouts, final_losses, s=100, alpha=0.6, edgecolors='black', linewidth=1, color='mediumpurple')
ax6.set_xlabel('Dropout Rate', fontsize=11, fontweight='bold')
ax6.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
ax6.set_title('Dropout Impact', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# 7. Embedding Dimension vs Validation Loss
ax7 = plt.subplot(3, 3, 7)
embedding_unique = sorted(set(embedding_dims))
emb_losses = [np.mean([final_losses[i] for i in range(len(trials)) if embedding_dims[i] == ed]) for ed in embedding_unique]
ax7.bar(range(len(embedding_unique)), emb_losses, color='gold', edgecolor='black', linewidth=1.5, alpha=0.7)
ax7.set_xticks(range(len(embedding_unique)))
ax7.set_xticklabels(embedding_unique)
ax7.set_xlabel('Embedding Dimension', fontsize=11, fontweight='bold')
ax7.set_ylabel('Average Validation Loss', fontsize=11, fontweight='bold')
ax7.set_title('Embedding Dimension Impact', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# 8. Training Time vs Validation Loss
ax8 = plt.subplot(3, 3, 8)
scatter2 = ax8.scatter(training_times, final_losses, c=final_losses, cmap='RdYlGn_r', s=120, alpha=0.6, edgecolors='black', linewidth=1)
ax8.set_xlabel('Training Time (seconds)', fontsize=11, fontweight='bold')
ax8.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
ax8.set_title('Training Time vs Performance', fontsize=12, fontweight='bold')
plt.colorbar(scatter2, ax=ax8, label='Val Loss')
ax8.grid(True, alpha=0.3)

# 9. Number of Parameters vs Validation Loss
ax9 = plt.subplot(3, 3, 9)
scatter3 = ax9.scatter(num_params, final_losses, c=num_params, cmap='plasma', s=120, alpha=0.6, edgecolors='black', linewidth=1)
ax9.set_xlabel('Number of Parameters', fontsize=11, fontweight='bold')
ax9.set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
ax9.set_title('Model Complexity vs Performance', fontsize=12, fontweight='bold')
plt.colorbar(scatter3, ax=ax9, label='Params')
ax9.grid(True, alpha=0.3)

# Overall title
fig.suptitle('Hyperparameter Search Results - 20 Trials Analysis', fontsize=16, fontweight='bold', y=0.995)

# Save figure
output_path = Path('hyperparameter_search_results/hyperparameter_analysis.png')
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Graph saved to: {output_path}")

# Also create a best parameters summary visualization
fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.axis('off')

# Create text summary with styling
best_params = results['best_params']
summary_text = f"""
HYPERPARAMETER OPTIMIZATION RESULTS

Best Trial: Trial 16
Best Validation Loss: {results['best_value']:.4f}
Total Optimization Time: {results['total_time']:.2f} seconds
Trials Completed: {results['n_trials']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OPTIMAL HYPERPARAMETERS:

  Learning Rate:        {best_params['learning_rate']:.6f}
  Batch Size:           {best_params['batch_size']}
  Hidden Size:          {best_params['hidden_size']}
  Number of Layers:     {best_params['num_layers']}
  Dropout:              {best_params['dropout']:.4f}
  Embedding Dimension:  {best_params['embedding_dim']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

KEY INSIGHTS:

• Best model uses 3 layers with 128 hidden units
• Moderate dropout (0.40) helps prevent overfitting
• Moderate learning rate (0.0012) ensures stable training
• Batch size of 32 provides good balance
• Embedding dimension of 96 captures enough information
"""

ax.text(0.1, 0.95, summary_text, transform=ax.transAxes, 
        fontsize=12, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, pad=1))

plt.tight_layout()
plt.savefig(Path('hyperparameter_search_results/best_parameters_summary.png'), dpi=300, bbox_inches='tight')
print(f"✓ Summary saved to: hyperparameter_search_results/best_parameters_summary.png")

print("\n✓ All graphs generated successfully!")
