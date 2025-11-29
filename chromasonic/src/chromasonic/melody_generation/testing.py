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
        
        # Ensure graphs directory exists
        path = Path(path)
        graphs_dir = path.parent / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        graph_path = graphs_dir / path.name
        
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
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Test plots saved to {graph_path}")
        
        # Generate description file
        desc_path = graph_path.with_suffix('.txt')
        self._save_test_description(desc_path, metrics)
        self.logger.info(f"Test description saved to {desc_path}")
    
    def _save_test_description(self, path: Path, metrics: Dict[str, float]):
        """Generate and save description file for test results."""
        from datetime import datetime
        
        # Get metric values
        accuracy = metrics.get('accuracy', 0.0)
        perplexity = metrics.get('perplexity', 0.0)
        diversity = metrics.get('diversity', 0.0)
        coherence = metrics.get('coherence', 0.0)
        musicality = metrics.get('musicality', 0.0)
        
        # Assess each metric
        def assess_metric(value, thresholds):
            """Assess metric based on thresholds (excellent, good, needs_improvement)."""
            if value >= thresholds[0]:
                return "Excellent"
            elif value >= thresholds[1]:
                return "Good"
            else:
                return "Needs Improvement"
        
        accuracy_assessment = assess_metric(accuracy, [0.7, 0.5])
        diversity_assessment = assess_metric(diversity, [0.7, 0.5])
        coherence_assessment = assess_metric(coherence, [0.7, 0.5])
        musicality_assessment = assess_metric(musicality, [0.7, 0.5])
        
        # Perplexity is inverse (lower is better)
        if perplexity < 10:
            perplexity_assessment = "Excellent"
        elif perplexity < 50:
            perplexity_assessment = "Good"
        else:
            perplexity_assessment = "Needs Improvement"
        
        overall_score = self.test_results.get('overall_score', 0.0)
        
        description = f"""TEST RESULTS ANALYSIS - {self.model_name}
{'=' * 80}

GRAPH OVERVIEW:
These visualizations show the comprehensive test performance of {self.model_name}
across multiple evaluation metrics.

{'=' * 80}
LEFT PLOT: DETAILED METRICS BAR CHART
{'=' * 80}

What it shows:
- Individual performance scores for each evaluation metric
- All metrics scaled to 0-1 range (higher is better, except perplexity)
- Bars show absolute performance on each dimension

Metric Breakdown:

1. ACCURACY: {accuracy:.4f} ({accuracy_assessment})
   - Measures: Next-note prediction correctness
   - Interpretation:
     ✓ >0.70: Model predicts notes very accurately
     ✓ 0.50-0.70: Reasonable prediction ability
     ✗ <0.50: Struggling to learn patterns
   Current: {accuracy_assessment} - Model {'excels at' if accuracy >= 0.7 else 'shows moderate' if accuracy >= 0.5 else 'struggles with'} predicting next notes

2. PERPLEXITY: {perplexity:.2f} ({perplexity_assessment})
   - Measures: Model confidence (lower is better)
   - Interpretation:
     ✓ <10: Very confident predictions
     ✓ 10-50: Moderate confidence
     ✗ >50: Uncertain/confused
   Current: {perplexity_assessment} - Model is {'very confident' if perplexity < 10 else 'moderately confident' if perplexity < 50 else 'uncertain'} in predictions

3. DIVERSITY: {diversity:.4f} ({diversity_assessment})
   - Measures: Variety in generated melodies
   - Interpretation:
     ✓ >0.70: High variety, creative outputs
     ✓ 0.50-0.70: Balanced repetition/variation
     ✗ <0.50: Too repetitive, limited creativity
   Current: {diversity_assessment} - Generated melodies are {'highly varied' if diversity >= 0.7 else 'moderately varied' if diversity >= 0.5 else 'quite repetitive'}

4. COHERENCE: {coherence:.4f} ({coherence_assessment})
   - Measures: Melodic smoothness and logical flow
   - Interpretation:
     ✓ >0.70: Smooth, singable melodies
     ✓ 0.50-0.70: Acceptable flow
     ✗ <0.50: Disjointed, random-sounding
   Current: {coherence_assessment} - Melodies are {'very smooth' if coherence >= 0.7 else 'moderately smooth' if coherence >= 0.5 else 'somewhat disjointed'}

5. MUSICALITY: {musicality:.4f} ({musicality_assessment})
   - Measures: Overall musical structure quality
   - Interpretation:
     ✓ >0.70: Strong musical structure (phrases, contours)
     ✓ 0.50-0.70: Some musical features present
     ✗ <0.50: Lacks musical organization
   Current: {musicality_assessment} - {'Strong' if musicality >= 0.7 else 'Moderate' if musicality >= 0.5 else 'Weak'} musical structure

{'=' * 80}
RIGHT PLOT: PERFORMANCE RADAR CHART
{'=' * 80}

What it shows:
- Multi-dimensional performance profile
- Larger coverage = better overall performance
- Shape shows strengths and weaknesses

Interpretation:
✓ Balanced polygon: Well-rounded model
✓ Large area: Strong overall performance
✗ Spiky shape: Imbalanced (strong in some areas, weak in others)
✗ Small area: Weak overall performance

Current Profile:
- Area coverage: {'Large' if overall_score > 0.7 else 'Medium' if overall_score > 0.5 else 'Small'}
- Balance: {'Well-balanced' if max(metrics.values()) - min(metrics.values()) < 0.3 else 'Some imbalance detected'}

{'=' * 80}
OVERALL ASSESSMENT
{'=' * 80}

Overall Score: {overall_score:.4f}

Performance Rating:
"""
        
        if overall_score >= 0.7:
            rating = "EXCELLENT - Model performs well across all metrics"
        elif overall_score >= 0.5:
            rating = "GOOD - Model shows solid performance with room for improvement"
        else:
            rating = "NEEDS IMPROVEMENT - Model requires optimization"
        
        description += rating + "\n\n"
        
        # Identify strengths and weaknesses
        description += "Strengths:\n"
        strengths = [k for k, v in metrics.items() if isinstance(v, float) and v >= 0.6]
        if strengths:
            for strength in strengths:
                description += f"  ✓ {strength.capitalize()}: {metrics[strength]:.4f}\n"
        else:
            description += "  • No standout strengths identified\n"
        
        description += "\nWeaknesses:\n"
        weaknesses = [k for k, v in metrics.items() if isinstance(v, float) and v < 0.5]
        if weaknesses:
            for weakness in weaknesses:
                description += f"  ✗ {weakness.capitalize()}: {metrics[weakness]:.4f}\n"
        else:
            description += "  • No major weaknesses identified\n"
        
        # Recommendations
        description += f"""\n{'=' * 80}
RECOMMENDATIONS
{'=' * 80}

"""
        
        recommendations = []
        
        if accuracy < 0.5:
            recommendations.append("• Improve accuracy: Train longer or use more training data")
            recommendations.append("• Consider increasing model capacity (more layers/units)")
        
        if diversity < 0.5:
            recommendations.append("• Increase diversity: Add temperature/sampling randomness")
            recommendations.append("• Reduce repetition in training data")
        
        if coherence < 0.5:
            recommendations.append("• Improve coherence: Add constraints for stepwise motion")
            recommendations.append("• Train on more musical data with better structure")
        
        if musicality < 0.5:
            recommendations.append("• Enhance musicality: Incorporate music theory rules")
            recommendations.append("• Add phrase structure constraints during generation")
        
        if perplexity > 50:
            recommendations.append("• Reduce perplexity: Model needs more training")
            recommendations.append("• Check training data quality and variety")
        
        if not recommendations:
            recommendations.append("• Model performs well - ready for production use")
            recommendations.append("• Consider A/B testing with other models")
            recommendations.append("• Collect user feedback for further refinement")
        
        description += "\n".join(recommendations)
        
        description += f"""\n
{'=' * 80}
COMPARISON WITH BASELINES
{'=' * 80}

Expected Performance Ranges:

• Markov Chain Baseline: 0.3-0.5 overall
• LSTM Models: 0.5-0.7 overall  
• Transformer Models: 0.6-0.8 overall
• Human Composers: 0.8-1.0 overall

Your Model ({self.model_name}): {overall_score:.4f}

{'=' * 80}
NEXT STEPS
{'=' * 80}

1. Compare with other model architectures
2. Listen to generated samples qualitatively
3. Conduct user studies for subjective evaluation
4. If performance is good, deploy to production
5. If performance is poor, iterate on architecture/training

{'=' * 80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
"""
        
        with open(path, 'w') as f:
            f.write(description)
    
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
        # Ensure graphs directory exists
        path = Path(path)
        graphs_dir = path.parent / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        graph_path = graphs_dir / path.name
        
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
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Comparison plot saved to {graph_path}")
        
        # Generate description file
        desc_path = graph_path.with_suffix('.txt')
        self._save_comparison_description(desc_path)
        self.logger.info(f"Comparison description saved to {desc_path}")
    
    def _save_comparison_description(self, path: Path):
        """Generate and save description file for model comparison."""
        from datetime import datetime
        
        model_names = list(self.comparison_results.keys())
        
        if not model_names:
            return
        
        # Find best and worst models
        best_model = max(model_names, key=lambda x: self.comparison_results[x]['overall_score'])
        worst_model = min(model_names, key=lambda x: self.comparison_results[x]['overall_score'])
        
        best_score = self.comparison_results[best_model]['overall_score']
        worst_score = self.comparison_results[worst_model]['overall_score']
        
        description = f"""MODEL COMPARISON ANALYSIS
{'=' * 80}

GRAPH OVERVIEW:
This visualization compares the performance of {len(model_names)} models
side-by-side across multiple evaluation metrics.

Models Compared:
"""
        
        for i, name in enumerate(model_names, 1):
            score = self.comparison_results[name]['overall_score']
            description += f"  {i}. {name}: {score:.4f}\n"
        
        description += f"""\n{'=' * 80}
LEFT PLOT: DETAILED METRIC COMPARISON
{'=' * 80}

What it shows:
- Grouped bar chart showing each model's performance on each metric
- Each color represents a different model
- Bars side-by-side allow direct comparison

How to interpret:
✓ Taller bars are better (except for perplexity - lower is better)
✓ Look for consistency across metrics (balanced performance)
✓ Large differences indicate clear winner/loser on that metric

Key Findings:

"""
        
        # Analyze each metric
        metrics = list(self.comparison_results[model_names[0]]['metrics'].keys())
        numeric_metrics = [
            m for m in metrics 
            if isinstance(self.comparison_results[model_names[0]]['metrics'][m], (int, float))
        ]
        
        for metric in numeric_metrics:
            values = {name: self.comparison_results[name]['metrics'][metric] 
                     for name in model_names}
            best_on_metric = max(values, key=values.get)
            worst_on_metric = min(values, key=values.get)
            
            description += f"\n{metric.upper()}:\n"
            description += f"  Best: {best_on_metric} ({values[best_on_metric]:.4f})\n"
            description += f"  Worst: {worst_on_metric} ({values[worst_on_metric]:.4f})\n"
            description += f"  Spread: {values[best_on_metric] - values[worst_on_metric]:.4f}\n"
        
        description += f"""\n{'=' * 80}
RIGHT PLOT: OVERALL SCORES COMPARISON
{'=' * 80}

What it shows:
- Single overall score for each model (average of all metrics)
- Color-coded bars for visual distinction
- Value labels on top of each bar

How to interpret:
✓ Higher score = better overall performance
✓ Score > 0.7: Excellent model
✓ Score 0.5-0.7: Good model
✗ Score < 0.5: Needs improvement

Rankings:

"""
        
        # Sort by score
        sorted_models = sorted(model_names, 
                             key=lambda x: self.comparison_results[x]['overall_score'], 
                             reverse=True)
        
        for i, name in enumerate(sorted_models, 1):
            score = self.comparison_results[name]['overall_score']
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            description += f"  {medal} {i}. {name}: {score:.4f}\n"
        
        description += f"""\n{'=' * 80}
DETAILED ANALYSIS
{'=' * 80}

Best Model: {best_model} (Score: {best_score:.4f})
"""
        
        # Analyze why best model is best
        best_strengths = []
        for metric in numeric_metrics:
            values = {name: self.comparison_results[name]['metrics'][metric] 
                     for name in model_names}
            if self.comparison_results[best_model]['metrics'][metric] == max(values.values()):
                best_strengths.append(metric)
        
        description += f"\nStrengths of {best_model}:\n"
        if best_strengths:
            for strength in best_strengths:
                value = self.comparison_results[best_model]['metrics'][strength]
                description += f"  ✓ Best {strength}: {value:.4f}\n"
        else:
            description += "  • Consistently good across all metrics\n"
        
        description += f"\nWorst Model: {worst_model} (Score: {worst_score:.4f})\n"
        
        # Analyze weaknesses
        worst_weaknesses = []
        for metric in numeric_metrics:
            values = {name: self.comparison_results[name]['metrics'][metric] 
                     for name in model_names}
            if self.comparison_results[worst_model]['metrics'][metric] == min(values.values()):
                worst_weaknesses.append(metric)
        
        description += f"\nWeaknesses of {worst_model}:\n"
        if worst_weaknesses:
            for weakness in worst_weaknesses:
                value = self.comparison_results[worst_model]['metrics'][weakness]
                description += f"  ✗ Worst {weakness}: {value:.4f}\n"
        else:
            description += "  • No single metric is worst, but overall score is lowest\n"
        
        # Performance gap analysis
        gap = best_score - worst_score
        description += f"""\n{'=' * 80}
PERFORMANCE GAP ANALYSIS
{'=' * 80}

Score Difference: {gap:.4f}

"""
        
        if gap < 0.1:
            description += "Interpretation: Models perform very similarly\n"
            description += "  • Differences may not be statistically significant\n"
            description += "  • Consider other factors (speed, complexity) for selection\n"
            description += "  • A/B testing may be needed to choose\n"
        elif gap < 0.2:
            description += "Interpretation: Moderate performance difference\n"
            description += f"  • {best_model} has a clear but not overwhelming advantage\n"
            description += "  • Both models could be viable for production\n"
        else:
            description += "Interpretation: Significant performance difference\n"
            description += f"  • {best_model} is clearly superior\n"
            description += f"  • {worst_model} needs substantial improvement\n"
            description += "  • Strong recommendation to use best model\n"
        
        description += f"""\n{'=' * 80}
RECOMMENDATIONS
{'=' * 80}

Production Deployment:
  → Use: {best_model}
  → Reason: Highest overall score ({best_score:.4f})

Model Selection Criteria:
"""
        
        # Provide selection criteria
        if gap > 0.2:
            description += f"  • Clear winner: Deploy {best_model}\n"
        else:
            description += "  • Close competition: Consider other factors:\n"
            description += "    - Inference speed\n"
            description += "    - Model size and memory requirements\n"
            description += "    - Training/maintenance costs\n"
            description += "    - Specific use case requirements\n"
        
        description += f"""\nFurther Investigation:
  1. Listen to generated samples from each model qualitatively
  2. Conduct user studies for subjective preferences
  3. Test on diverse image inputs (landscapes, portraits, abstract)
  4. Measure inference time and resource usage
  5. Evaluate robustness to edge cases

Next Steps:
  • If satisfied: Deploy {best_model} to production
  • If not satisfied: Continue model development and training
  • Consider ensemble methods combining multiple models
  • Collect real user feedback and iterate

{'=' * 80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
"""
        
        with open(path, 'w') as f:
            f.write(description)
    
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
