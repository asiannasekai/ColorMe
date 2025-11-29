"""
Testing Module for Melody Generation Models
Provides comprehensive testing, evaluation, and comparison tools.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import json

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ModelTester:
    """Comprehensive testing for melody generation models."""
    
    def __init__(self, model: Any, model_name: str = "model"):
        """
        Initialize model tester.
        
        Args:
            model: Model to test
            model_name: Name of the model
        """
        self.model = model
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.test_results = {}
    
    def run_comprehensive_test(
        self,
        test_data: List[List[int]],
        metrics: Optional[List[str]] = None,
        save_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive testing suite.
        
        Args:
            test_data: Test sequences
            metrics: List of metrics to compute
            save_dir: Directory to save results
            
        Returns:
            Dictionary of test results
        """
        if metrics is None:
            metrics = [
                'accuracy',
                'perplexity',
                'diversity',
                'coherence',
                'musicality'
            ]
        
        results = {
            'model_name': self.model_name,
            'num_test_sequences': len(test_data),
            'metrics': {}
        }
        
        self.logger.info(f"Running comprehensive test for {self.model_name}")
        self.logger.info(f"Test sequences: {len(test_data)}")
        
        # Run each metric
        if 'accuracy' in metrics:
            results['metrics']['accuracy'] = self.test_prediction_accuracy(test_data)
        
        if 'perplexity' in metrics:
            results['metrics']['perplexity'] = self.test_perplexity(test_data)
        
        if 'diversity' in metrics:
            results['metrics']['diversity'] = self.test_diversity(test_data)
        
        if 'coherence' in metrics:
            results['metrics']['coherence'] = self.test_coherence(test_data)
        
        if 'musicality' in metrics:
            results['metrics']['musicality'] = self.test_musicality(test_data)
        
        # Calculate overall score
        results['overall_score'] = np.mean([
            v for v in results['metrics'].values() if isinstance(v, (int, float))
        ])
        
        self.test_results = results
        
        # Save results
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            self._save_results(save_dir / f"{self.model_name}_test_results.json")
            self._plot_results(save_dir / f"{self.model_name}_test_plots.png")
        
        self._print_results()
        
        return results
    
    def test_prediction_accuracy(self, test_data: List[List[int]]) -> float:
        """
        Test next-note prediction accuracy.
        
        Args:
            test_data: Test sequences
            
        Returns:
            Accuracy score (0-1)
        """
        correct = 0
        total = 0
        
        for sequence in test_data:
            if len(sequence) < 2:
                continue
            
            # Test prediction at each position
            for i in range(len(sequence) - 1):
                context = sequence[:i+1]
                true_next = sequence[i+1]
                
                # Get model prediction
                predicted_next = self._predict_next_note(context)
                
                if predicted_next == true_next:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        self.logger.info(f"Prediction Accuracy: {accuracy:.2%}")
        
        return accuracy
    
    def test_perplexity(self, test_data: List[List[int]]) -> float:
        """
        Compute perplexity on test data.
        
        Args:
            test_data: Test sequences
            
        Returns:
            Perplexity score (lower is better)
        """
        total_log_prob = 0.0
        total_notes = 0
        
        for sequence in test_data:
            if len(sequence) < 2:
                continue
            
            for i in range(len(sequence) - 1):
                context = sequence[:i+1]
                true_next = sequence[i+1]
                
                # Get probability of true next note
                prob = self._get_next_note_probability(context, true_next)
                
                if prob > 0:
                    total_log_prob += np.log(prob)
                    total_notes += 1
        
        # Calculate perplexity
        avg_log_prob = total_log_prob / total_notes if total_notes > 0 else -10
        perplexity = np.exp(-avg_log_prob)
        
        self.logger.info(f"Perplexity: {perplexity:.2f}")
        
        return perplexity
    
    def test_diversity(self, test_data: List[List[int]], num_samples: int = 100) -> float:
        """
        Test diversity of generated sequences.
        
        Args:
            test_data: Test sequences for seeding
            num_samples: Number of sequences to generate
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        generated_sequences = []
        
        # Generate sequences
        for i in range(num_samples):
            if i < len(test_data):
                seed = test_data[i][:8]  # Use first 8 notes as seed
            else:
                seed = [np.random.randint(0, 88)]
            
            generated = self._generate_sequence(seed, length=16)
            generated_sequences.append(tuple(generated))
        
        # Calculate uniqueness
        unique_sequences = len(set(generated_sequences))
        uniqueness = unique_sequences / num_samples
        
        # Calculate vocabulary usage
        all_notes = [note for seq in generated_sequences for note in seq]
        unique_notes = len(set(all_notes))
        vocab_usage = unique_notes / 88  # Assuming 88 possible notes
        
        # Calculate interval variety
        all_intervals = []
        for seq in generated_sequences:
            intervals = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
            all_intervals.extend(intervals)
        
        unique_intervals = len(set(all_intervals))
        interval_diversity = min(1.0, unique_intervals / 20)  # Max ~20 common intervals
        
        # Combined diversity score
        diversity = (uniqueness + vocab_usage + interval_diversity) / 3
        
        self.logger.info(f"Diversity Score: {diversity:.2%}")
        self.logger.info(f"  - Unique sequences: {unique_sequences}/{num_samples}")
        self.logger.info(f"  - Vocabulary usage: {unique_notes}/88")
        self.logger.info(f"  - Unique intervals: {unique_intervals}")
        
        return diversity
    
    def test_coherence(self, test_data: List[List[int]]) -> float:
        """
        Test melodic coherence and smoothness.
        
        Args:
            test_data: Test sequences
            
        Returns:
            Coherence score (0-1)
        """
        coherence_scores = []
        
        for sequence in test_data:
            if len(sequence) < 4:
                continue
            
            # Calculate interval smoothness
            intervals = [abs(sequence[i+1] - sequence[i]) for i in range(len(sequence)-1)]
            
            # Prefer small intervals (stepwise motion)
            smooth_intervals = sum(1 for interval in intervals if interval <= 2)
            smoothness = smooth_intervals / len(intervals)
            
            # Penalize large jumps
            large_jumps = sum(1 for interval in intervals if interval > 5)
            jump_penalty = large_jumps / len(intervals)
            
            # Check for monotonic sections (lack of direction changes)
            direction_changes = 0
            for i in range(1, len(intervals)):
                if (intervals[i] > 0) != (intervals[i-1] > 0):
                    direction_changes += 1
            
            direction_variety = direction_changes / (len(intervals) - 1) if len(intervals) > 1 else 0
            
            # Combined coherence
            seq_coherence = (smoothness * 0.5 + 
                           (1 - jump_penalty) * 0.3 + 
                           direction_variety * 0.2)
            
            coherence_scores.append(seq_coherence)
        
        coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        
        self.logger.info(f"Coherence Score: {coherence:.2%}")
        
        return coherence
    
    def test_musicality(self, test_data: List[List[int]]) -> float:
        """
        Test overall musicality including rhythm and phrasing.
        
        Args:
            test_data: Test sequences
            
        Returns:
            Musicality score (0-1)
        """
        musicality_scores = []
        
        for sequence in test_data:
            if len(sequence) < 8:
                continue
            
            score = 0.0
            
            # 1. Phrase structure (4 or 8 note phrases)
            phrase_lengths = [4, 8]
            has_phrase_structure = any(len(sequence) % length == 0 for length in phrase_lengths)
            if has_phrase_structure:
                score += 0.2
            
            # 2. Contour shape (arch or wave patterns)
            mid_point = len(sequence) // 2
            first_half_avg = np.mean(sequence[:mid_point])
            second_half_avg = np.mean(sequence[mid_point:])
            has_contour = abs(first_half_avg - second_half_avg) > 2
            if has_contour:
                score += 0.2
            
            # 3. Repetition and variation
            note_counts = Counter(sequence)
            max_repetition = max(note_counts.values()) / len(sequence)
            good_repetition = 0.2 <= max_repetition <= 0.4  # Some but not too much
            if good_repetition:
                score += 0.2
            
            # 4. Range appropriateness (1-2 octaves)
            pitch_range = max(sequence) - min(sequence)
            appropriate_range = 7 <= pitch_range <= 14  # 7-14 semitones
            if appropriate_range:
                score += 0.2
            
            # 5. Ending resolution (tendency to return near starting note)
            start_note = sequence[0]
            end_note = sequence[-1]
            resolves = abs(end_note - start_note) <= 3
            if resolves:
                score += 0.2
            
            musicality_scores.append(score)
        
        musicality = np.mean(musicality_scores) if musicality_scores else 0.0
        
        self.logger.info(f"Musicality Score: {musicality:.2%}")
        
        return musicality
    
    def _predict_next_note(self, context: List[int]) -> int:
        """Predict next note given context."""
        # This is a placeholder - implement based on model type
        if hasattr(self.model, 'predict_next'):
            return self.model.predict_next(context)
        elif hasattr(self.model, 'generate'):
            # For generative models
            generated = self.model.generate(context, length=1)
            return generated[-1] if generated else 0
        else:
            # Fallback: return most common note in context
            return Counter(context).most_common(1)[0][0] if context else 0
    
    def _get_next_note_probability(self, context: List[int], next_note: int) -> float:
        """Get probability of next note given context."""
        # Placeholder - implement based on model type
        if hasattr(self.model, 'get_probability'):
            return self.model.get_probability(context, next_note)
        else:
            # Fallback: uniform probability
            return 1.0 / 88
    
    def _generate_sequence(self, seed: List[int], length: int) -> List[int]:
        """Generate a sequence from seed."""
        if hasattr(self.model, 'generate'):
            return self.model.generate(seed, length=length)
        else:
            # Fallback: random walk
            sequence = seed.copy()
            for _ in range(length - len(seed)):
                next_note = max(0, min(87, sequence[-1] + np.random.choice([-1, 0, 1])))
                sequence.append(next_note)
            return sequence
    
    def _save_results(self, path: Path):
        """Save test results to JSON."""
        with open(path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        self.logger.info(f"Test results saved to {path}")
    
    def _plot_results(self, path: Path):
        """Plot test results."""
        metrics = self.test_results.get('metrics', {})
        
        if not metrics:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart of metrics
        metric_names = list(metrics.keys())
        metric_values = [metrics[k] for k in metric_names if isinstance(metrics[k], (int, float))]
        valid_names = [k for k in metric_names if isinstance(metrics[k], (int, float))]
        
        ax1.bar(valid_names, metric_values, color='steelblue', alpha=0.7)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title(f'{self.model_name} - Test Metrics', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Radar chart
        if len(valid_names) >= 3:
            angles = np.linspace(0, 2 * np.pi, len(valid_names), endpoint=False).tolist()
            values = metric_values + [metric_values[0]]  # Complete the circle
            angles += angles[:1]
            
            ax2 = plt.subplot(122, projection='polar')
            ax2.plot(angles, values, 'o-', linewidth=2, color='steelblue')
            ax2.fill(angles, values, alpha=0.25, color='steelblue')
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(valid_names)
            ax2.set_ylim(0, 1)
            ax2.set_title(f'{self.model_name} - Performance Radar', 
                         fontsize=14, fontweight='bold', pad=20)
            ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Test plots saved to {path}")
    
    def _print_results(self):
        """Print formatted test results."""
        print("\n" + "="*60)
        print(f"Test Results for {self.model_name}")
        print("="*60)
        
        metrics = self.test_results.get('metrics', {})
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric_name.capitalize():20s}: {value:.4f}")
            else:
                print(f"{metric_name.capitalize():20s}: {value}")
        
        print("-"*60)
        print(f"{'Overall Score':20s}: {self.test_results.get('overall_score', 0):.4f}")
        print("="*60 + "\n")


class ModelComparison:
    """Compare multiple models on the same test set."""
    
    def __init__(self):
        """Initialize model comparison."""
        self.logger = logging.getLogger(__name__)
        self.comparison_results = {}
    
    def compare_models(
        self,
        models: Dict[str, Any],
        test_data: List[List[int]],
        metrics: Optional[List[str]] = None,
        save_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary of {name: model}
            test_data: Test sequences
            metrics: Metrics to compute
            save_dir: Directory to save comparison results
            
        Returns:
            Comparison results
        """
        self.logger.info(f"Comparing {len(models)} models")
        
        results = {}
        
        for model_name, model in models.items():
            self.logger.info(f"\nTesting {model_name}...")
            tester = ModelTester(model, model_name)
            results[model_name] = tester.run_comprehensive_test(
                test_data, metrics, save_dir=None  # Don't save individual results yet
            )
        
        self.comparison_results = results
        
        # Save comparison
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            self._save_comparison(save_dir / "model_comparison.json")
            self._plot_comparison(save_dir / "model_comparison.png")
        
        self._print_comparison()
        
        return results
    
    def _save_comparison(self, path: Path):
        """Save comparison results."""
        with open(path, 'w') as f:
            json.dump(self.comparison_results, f, indent=2)
        
        self.logger.info(f"Comparison results saved to {path}")
    
    def _plot_comparison(self, path: Path):
        """Plot model comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prepare data
        model_names = list(self.comparison_results.keys())
        metrics_names = list(self.comparison_results[model_names[0]]['metrics'].keys())
        
        # Filter numeric metrics
        numeric_metrics = [
            m for m in metrics_names 
            if isinstance(self.comparison_results[model_names[0]]['metrics'][m], (int, float))
        ]
        
        # Grouped bar chart
        x = np.arange(len(numeric_metrics))
        width = 0.8 / len(model_names)
        
        for i, model_name in enumerate(model_names):
            values = [self.comparison_results[model_name]['metrics'][m] for m in numeric_metrics]
            axes[0].bar(x + i * width, values, width, label=model_name, alpha=0.8)
        
        axes[0].set_xlabel('Metrics', fontsize=12)
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].set_title('Model Comparison - Detailed Metrics', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x + width * (len(model_names) - 1) / 2)
        axes[0].set_xticklabels(numeric_metrics, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim(0, 1.1)
        
        # Overall scores
        overall_scores = [self.comparison_results[name]['overall_score'] for name in model_names]
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        
        axes[1].bar(model_names, overall_scores, color=colors, alpha=0.7)
        axes[1].set_ylabel('Overall Score', fontsize=12)
        axes[1].set_title('Model Comparison - Overall Scores', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim(0, 1.1)
        
        # Add value labels on bars
        for i, (name, score) in enumerate(zip(model_names, overall_scores)):
            axes[1].text(i, score + 0.02, f'{score:.3f}', 
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Comparison plot saved to {path}")
    
    def _print_comparison(self):
        """Print formatted comparison results."""
        print("\n" + "="*80)
        print("Model Comparison Results")
        print("="*80)
        
        # Print header
        model_names = list(self.comparison_results.keys())
        header = f"{'Metric':<20s} " + " ".join([f"{name:>15s}" for name in model_names])
        print(header)
        print("-"*80)
        
        # Print each metric
        if model_names:
            metrics = self.comparison_results[model_names[0]]['metrics'].keys()
            
            for metric in metrics:
                values = []
                for model_name in model_names:
                    value = self.comparison_results[model_name]['metrics'][metric]
                    if isinstance(value, float):
                        values.append(f"{value:>15.4f}")
                    else:
                        values.append(f"{value:>15}")
                
                print(f"{metric:<20s} " + " ".join(values))
        
        print("-"*80)
        
        # Print overall scores
        overall_line = f"{'Overall Score':<20s} "
        for model_name in model_names:
            score = self.comparison_results[model_name]['overall_score']
            overall_line += f"{score:>15.4f}"
        
        print(overall_line)
        print("="*80 + "\n")
        
        # Print winner
        best_model = max(model_names, 
                        key=lambda x: self.comparison_results[x]['overall_score'])
        best_score = self.comparison_results[best_model]['overall_score']
        
        print(f"🏆 Best Model: {best_model} (Score: {best_score:.4f})\n")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Model Testing Module")
    print("This module provides comprehensive testing for melody generation models.")
    print("\nExample usage:")
    print("""
    from chromasonic.melody_generation.testing import ModelTester
    
    # Test a single model
    tester = ModelTester(your_model, "MyModel")
    results = tester.run_comprehensive_test(
        test_data=test_sequences,
        save_dir=Path("./test_results")
    )
    
    # Compare multiple models
    from chromasonic.melody_generation.testing import ModelComparison
    
    comparison = ModelComparison()
    results = comparison.compare_models(
        models={"Markov": markov_model, "LSTM": lstm_model},
        test_data=test_sequences,
        save_dir=Path("./comparison_results")
    )
    """)
