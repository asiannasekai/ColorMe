"""
Hyperparameter Search Module for Melody Generation Models
Provides automated hyperparameter optimization using Optuna framework.
"""

import logging
import time
import json
from typing import Dict, Any, Optional, Tuple, Callable
from pathlib import Path
import numpy as np
from datetime import datetime

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not installed. Install with: pip install optuna")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from .training import ModelTrainer, TrainingHistory, MelodyDataset
    HAS_TRAINING = True
except ImportError:
    HAS_TRAINING = False


class HyperparameterSearchSpace:
    """Defines the hyperparameter search space for model tuning."""
    
    def __init__(
        self,
        learning_rate_range: Tuple[float, float] = (1e-5, 1e-2),
        batch_size_options: list = [16, 32, 64, 128],
        hidden_size_range: Tuple[int, int] = (64, 512),
        num_layers_range: Tuple[int, int] = (1, 4),
        dropout_range: Tuple[float, float] = (0.0, 0.5),
        embedding_dim_range: Tuple[int, int] = (32, 256),
        num_epochs: int = 20
    ):
        """
        Initialize search space.
        
        Args:
            learning_rate_range: Min and max learning rate to search
            batch_size_options: List of batch sizes to try
            hidden_size_range: Min and max hidden layer size
            num_layers_range: Min and max number of layers
            dropout_range: Min and max dropout rate
            embedding_dim_range: Min and max embedding dimension
            num_epochs: Number of epochs for each trial
        """
        self.learning_rate_range = learning_rate_range
        self.batch_size_options = batch_size_options
        self.hidden_size_range = hidden_size_range
        self.num_layers_range = num_layers_range
        self.dropout_range = dropout_range
        self.embedding_dim_range = embedding_dim_range
        self.num_epochs = num_epochs
        
    def suggest_hyperparameters(self, trial: 'optuna.Trial') -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of hyperparameters
        """
        return {
            'learning_rate': trial.suggest_float(
                'learning_rate',
                self.learning_rate_range[0],
                self.learning_rate_range[1],
                log=True
            ),
            'batch_size': trial.suggest_categorical('batch_size', self.batch_size_options),
            'hidden_size': trial.suggest_int(
                'hidden_size',
                self.hidden_size_range[0],
                self.hidden_size_range[1],
                step=32
            ),
            'num_layers': trial.suggest_int(
                'num_layers',
                self.num_layers_range[0],
                self.num_layers_range[1]
            ),
            'dropout': trial.suggest_float(
                'dropout',
                self.dropout_range[0],
                self.dropout_range[1]
            ),
            'embedding_dim': trial.suggest_int(
                'embedding_dim',
                self.embedding_dim_range[0],
                self.embedding_dim_range[1],
                step=32
            ),
        }


class HyperparameterOptimizer:
    """Optimizes model hyperparameters using Optuna."""
    
    def __init__(
        self,
        model_class,
        search_space: HyperparameterSearchSpace,
        device: str = 'cpu',
        output_dir: Optional[Path] = None,
        verbose: bool = True
    ):
        """
        Initialize optimizer.
        
        Args:
            model_class: Model class to optimize (SimpleRNN, Transformer, etc.)
            search_space: HyperparameterSearchSpace instance
            device: 'cpu' or 'cuda'
            output_dir: Directory to save results
            verbose: Whether to print progress
        """
        if not HAS_OPTUNA:
            raise ImportError("Optuna is required for hyperparameter search")
        
        self.model_class = model_class
        self.search_space = search_space
        self.device = device
        self.output_dir = output_dir or Path('hyperparameter_search_results')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.logger = self._setup_logger()
        
        self.best_params = None
        self.best_value = float('inf')
        self.study = None
        self.trial_results = []
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging."""
        logger = logging.getLogger('HyperparameterOptimizer')
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        # Remove existing handlers
        logger.handlers = []
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO if self.verbose else logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger
    
    def objective(
        self,
        trial: 'optuna.Trial',
        train_data: list,
        val_data: list,
        vocab_size: int,
        sequence_length: int = 16
    ) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            train_data: Training sequences
            val_data: Validation sequences
            vocab_size: Size of vocabulary (number of notes)
            sequence_length: Length of input sequences
            
        Returns:
            Validation loss (to minimize)
        """
        try:
            # Suggest hyperparameters
            params = self.search_space.suggest_hyperparameters(trial)
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Trial {trial.number}: {params}")
            self.logger.info(f"{'='*60}")
            
            # Create model
            model = self.model_class(
                vocab_size=vocab_size,
                embedding_dim=params['embedding_dim'],
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout']
            )
            
            model = model.to(self.device)
            
            # Create trainer
            trainer = ModelTrainer(
                model,
                device=self.device,
                model_name=f"trial_{trial.number}"
            )
            
            # Train model
            start_time = time.time()
            history = trainer.train(
                train_data=train_data,
                val_data=val_data,
                epochs=self.search_space.num_epochs,
                batch_size=params['batch_size'],
                learning_rate=params['learning_rate'],
                sequence_length=sequence_length,
                early_stopping_patience=5,
                checkpoint_dir=None,
                lr_scheduler='cosine'
            )
            training_time = time.time() - start_time
            
            # Get final validation loss
            final_val_loss = history.val_losses[-1]
            
            # Store results
            result = {
                'trial': trial.number,
                'params': params,
                'final_val_loss': final_val_loss,
                'final_train_loss': history.train_losses[-1],
                'best_val_loss': history.best_val_loss,
                'best_epoch': history.best_epoch,
                'training_time': training_time,
                'num_parameters': sum(p.numel() for p in model.parameters())
            }
            self.trial_results.append(result)
            
            # Log results
            self.logger.info(f"Trial {trial.number} Results:")
            self.logger.info(f"  Final Val Loss: {final_val_loss:.4f}")
            self.logger.info(f"  Best Val Loss: {history.best_val_loss:.4f} (epoch {history.best_epoch})")
            self.logger.info(f"  Training Time: {training_time:.2f}s")
            self.logger.info(f"  Model Parameters: {result['num_parameters']:,}")
            
            # Update best parameters
            if final_val_loss < self.best_value:
                self.best_value = final_val_loss
                self.best_params = params
                self.logger.info(f"  ✓ New best! Validation loss: {final_val_loss:.4f}")
            
            return final_val_loss
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed with error: {str(e)}")
            return float('inf')
    
    def optimize(
        self,
        train_data: list,
        val_data: list,
        vocab_size: int,
        sequence_length: int = 16,
        n_trials: int = 20,
        timeout: Optional[float] = None,
        study_name: str = "melody_hp_search"
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            train_data: Training sequences
            val_data: Validation sequences
            vocab_size: Size of vocabulary
            sequence_length: Length of input sequences
            n_trials: Number of trials to run
            timeout: Optional timeout in seconds
            study_name: Name for the Optuna study
            
        Returns:
            Dictionary with best parameters and results
        """
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Starting Hyperparameter Optimization")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Number of trials: {n_trials}")
        self.logger.info(f"Timeout: {timeout}s" if timeout else "No timeout")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"{'='*80}\n")
        
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)
        
        self.study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            study_name=study_name
        )
        
        # Run optimization
        start_time = time.time()
        self.study.optimize(
            lambda trial: self.objective(
                trial,
                train_data,
                val_data,
                vocab_size,
                sequence_length
            ),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=self.verbose
        )
        total_time = time.time() - start_time
        
        # Set default best params if none were found
        if self.best_params is None:
            self.best_params = self.search_space.suggest_hyperparameters(self.study.best_trial)
        
        # Log summary
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Optimization Complete")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Total Time: {total_time:.2f}s")
        self.logger.info(f"Trials Completed: {len(self.study.trials)}")
        self.logger.info(f"Best Validation Loss: {self.best_value:.4f}")
        self.logger.info(f"\nBest Hyperparameters:")
        for param, value in self.best_params.items():
            self.logger.info(f"  {param}: {value}")
        self.logger.info(f"{'='*80}\n")
        
        # Save results
        self._save_results(total_time)
        
        return {
            'best_params': self.best_params,
            'best_value': self.best_value,
            'total_time': total_time,
            'n_trials': len(self.study.trials),
            'trial_results': self.trial_results
        }
    
    def _save_results(self, total_time: float):
        """Save optimization results to file."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_time': total_time,
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(self.study.trials),
            'trial_results': self.trial_results
        }
        
        # Save as JSON
        results_path = self.output_dir / 'optimization_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Results saved to {results_path}")
        
        # Save best parameters as config
        config_path = self.output_dir / 'best_hyperparameters.json'
        with open(config_path, 'w') as f:
            json.dump(self.best_params, f, indent=2, default=str)
        self.logger.info(f"Best hyperparameters saved to {config_path}")
        
        # Generate summary report
        self._generate_report()
    
    def _generate_report(self):
        """Generate a summary report of the optimization."""
        report_path = self.output_dir / 'optimization_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("HYPERPARAMETER OPTIMIZATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Number of Trials: {len(self.study.trials)}\n")
            f.write(f"Best Validation Loss: {self.best_value:.4f}\n\n")
            
            f.write("BEST HYPERPARAMETERS:\n")
            f.write("-" * 80 + "\n")
            for param, value in self.best_params.items():
                f.write(f"  {param:20s}: {value}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("TOP 10 TRIALS:\n")
            f.write("-" * 80 + "\n")
            
            sorted_results = sorted(
                self.trial_results,
                key=lambda x: x['final_val_loss']
            )[:10]
            
            for i, result in enumerate(sorted_results, 1):
                f.write(f"\nRank {i}: Trial {result['trial']}\n")
                f.write(f"  Val Loss: {result['final_val_loss']:.4f}\n")
                f.write(f"  Train Loss: {result['final_train_loss']:.4f}\n")
                f.write(f"  Best Epoch: {result['best_epoch']}\n")
                f.write(f"  Training Time: {result['training_time']:.2f}s\n")
                f.write(f"  Parameters: {result['num_parameters']:,}\n")
                f.write(f"  Hyperparameters:\n")
                for param, value in result['params'].items():
                    f.write(f"    {param}: {value}\n")
        
        self.logger.info(f"Report saved to {report_path}")
    
    def plot_optimization_history(self, save_path: Optional[Path] = None):
        """
        Plot optimization history.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.study:
            self.logger.warning("No optimization study to plot")
            return
        
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.warning("matplotlib not available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Hyperparameter Optimization History', fontsize=16, fontweight='bold')
        
        # Plot 1: Trials over time
        losses = [r['final_val_loss'] for r in self.trial_results]
        axes[0, 0].plot(losses, marker='o', linestyle='-', alpha=0.7)
        axes[0, 0].axhline(y=min(losses), color='r', linestyle='--', label='Best')
        axes[0, 0].set_xlabel('Trial')
        axes[0, 0].set_ylabel('Validation Loss')
        axes[0, 0].set_title('Validation Loss Over Trials')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Parameter importance (if available)
        if len(self.trial_results) > 10:
            learning_rates = [r['params']['learning_rate'] for r in self.trial_results]
            batch_sizes = [r['params']['batch_size'] for r in self.trial_results]
            
            axes[0, 1].scatter(learning_rates, losses, alpha=0.6)
            axes[0, 1].set_xlabel('Learning Rate')
            axes[0, 1].set_ylabel('Validation Loss')
            axes[0, 1].set_title('Learning Rate Impact')
            axes[0, 1].set_xscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Best loss over iterations
        best_losses = []
        current_best = float('inf')
        for result in self.trial_results:
            if result['final_val_loss'] < current_best:
                current_best = result['final_val_loss']
            best_losses.append(current_best)
        
        axes[1, 0].plot(best_losses, marker='s', linestyle='-', color='green', alpha=0.7)
        axes[1, 0].set_xlabel('Trial')
        axes[1, 0].set_ylabel('Best Validation Loss')
        axes[1, 0].set_title('Best Loss Found Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Hidden size vs validation loss
        hidden_sizes = [r['params']['hidden_size'] for r in self.trial_results]
        axes[1, 1].scatter(hidden_sizes, losses, alpha=0.6)
        axes[1, 1].set_xlabel('Hidden Size')
        axes[1, 1].set_ylabel('Validation Loss')
        axes[1, 1].set_title('Hidden Size Impact')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
