"""
Evaluation Metrics Module
Provides comprehensive evaluation metrics for assessing the quality of image-to-music conversion.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import Counter
import math


class MusicalQualityMetrics:
    """Evaluates musical quality of generated melodies."""
    
    def __init__(self):
        """Initialize musical quality evaluator."""
        self.logger = logging.getLogger(__name__)
    
    def evaluate_melody(self, melody: List[int], scale: List[int]) -> Dict[str, float]:
        """
        Evaluate musical quality of a melody.
        
        Args:
            melody: List of note indices in scale
            scale: Scale intervals
            
        Returns:
            Dictionary with quality metrics
        """
        
        metrics = {}
        
        # Melodic contour quality
        metrics['contour_smoothness'] = self._evaluate_contour_smoothness(melody)
        
        # Interval distribution
        metrics['interval_variety'] = self._evaluate_interval_variety(melody)
        
        # Scale adherence
        metrics['scale_adherence'] = self._evaluate_scale_adherence(melody, scale)
        
        # Rhythmic coherence (if rhythm data available)
        metrics['pitch_range'] = self._evaluate_pitch_range(melody)
        
        # Musical phrases
        metrics['phrase_structure'] = self._evaluate_phrase_structure(melody)
        
        # Repetition and variation
        metrics['repetition_balance'] = self._evaluate_repetition_balance(melody)
        
        # Overall musical quality
        metrics['overall_quality'] = np.mean([
            metrics['contour_smoothness'],
            metrics['interval_variety'],
            metrics['scale_adherence'],
            metrics['phrase_structure']
        ])
        
        return metrics
    
    def _evaluate_contour_smoothness(self, melody: List[int]) -> float:
        """Evaluate smoothness of melodic contour."""
        
        if len(melody) < 2:
            return 1.0
        
        # Calculate intervals between consecutive notes
        intervals = [abs(melody[i+1] - melody[i]) for i in range(len(melody)-1)]
        
        # Prefer smaller intervals (stepwise motion)
        small_intervals = sum(1 for interval in intervals if interval <= 2)
        smoothness = small_intervals / len(intervals)
        
        # Penalize large jumps
        large_jumps = sum(1 for interval in intervals if interval > 4)
        jump_penalty = large_jumps / len(intervals) * 0.3
        
        return max(0.0, smoothness - jump_penalty)
    
    def _evaluate_interval_variety(self, melody: List[int]) -> float:
        """Evaluate variety in melodic intervals."""
        
        if len(melody) < 2:
            return 0.5
        
        # Calculate intervals
        intervals = [melody[i+1] - melody[i] for i in range(len(melody)-1)]
        
        # Count unique intervals
        unique_intervals = len(set(intervals))
        max_possible = min(len(intervals), 7)  # Reasonable maximum
        
        variety = unique_intervals / max_possible
        
        # Bonus for balanced use of ascending/descending
        ascending = sum(1 for interval in intervals if interval > 0)
        descending = sum(1 for interval in intervals if interval < 0)
        
        if len(intervals) > 0:
            balance = 1.0 - abs(ascending - descending) / len(intervals)
            variety = (variety + balance) / 2.0
        
        return variety
    
    def _evaluate_scale_adherence(self, melody: List[int], scale: List[int]) -> float:
        """Evaluate how well melody adheres to scale."""
        
        if not melody or not scale:
            return 0.0
        
        # All notes should be valid scale indices
        valid_notes = sum(1 for note in melody if 0 <= note < len(scale))
        adherence = valid_notes / len(melody)
        
        return adherence
    
    def _evaluate_pitch_range(self, melody: List[int]) -> float:
        """Evaluate pitch range usage."""
        
        if not melody:
            return 0.0
        
        pitch_range = max(melody) - min(melody)
        
        # Optimal range is around 1-2 octaves (7-14 semitones)
        if 7 <= pitch_range <= 14:
            return 1.0
        elif pitch_range < 7:
            return pitch_range / 7.0
        else:
            return max(0.3, 1.0 - (pitch_range - 14) / 14.0)
    
    def _evaluate_phrase_structure(self, melody: List[int]) -> float:
        """Evaluate phrase structure and cadences."""
        
        if len(melody) < 8:
            return 0.5  # Too short to evaluate phrases
        
        # Simple phrase detection based on repeated patterns and returns to tonic
        phrase_score = 0.0
        
        # Check for phrase boundaries (every 4-8 notes)
        phrase_length = 4
        num_phrases = len(melody) // phrase_length
        
        if num_phrases >= 2:
            # Check if phrases have similar patterns or complement each other
            phrases = [melody[i*phrase_length:(i+1)*phrase_length] 
                      for i in range(num_phrases)]
            
            # Simple similarity check
            similarities = []
            for i in range(len(phrases)-1):
                similarity = self._calculate_phrase_similarity(phrases[i], phrases[i+1])
                similarities.append(similarity)
            
            # Good phrase structure has moderate similarity (not identical, not random)
            avg_similarity = np.mean(similarities)
            if 0.3 <= avg_similarity <= 0.7:
                phrase_score = 1.0
            else:
                phrase_score = 1.0 - abs(avg_similarity - 0.5) * 2
        
        return max(0.0, phrase_score)
    
    def _calculate_phrase_similarity(self, phrase1: List[int], phrase2: List[int]) -> float:
        """Calculate similarity between two phrases."""
        
        if len(phrase1) != len(phrase2):
            return 0.0
        
        # Check interval patterns rather than absolute pitches
        intervals1 = [phrase1[i+1] - phrase1[i] for i in range(len(phrase1)-1)]
        intervals2 = [phrase2[i+1] - phrase2[i] for i in range(len(phrase2)-1)]
        
        if not intervals1:
            return 1.0 if phrase1 == phrase2 else 0.0
        
        matches = sum(1 for i1, i2 in zip(intervals1, intervals2) if abs(i1 - i2) <= 1)
        similarity = matches / len(intervals1)
        
        return similarity
    
    def _evaluate_repetition_balance(self, melody: List[int]) -> float:
        """Evaluate balance between repetition and variation."""
        
        if len(melody) < 4:
            return 0.5
        
        # Count note frequencies
        note_counts = Counter(melody)
        
        # Calculate entropy (measure of variety)
        total_notes = len(melody)
        entropy = 0.0
        for count in note_counts.values():
            probability = count / total_notes
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # Normalize entropy (0 to 1)
        max_entropy = math.log2(len(note_counts))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 0.0
        
        # Optimal balance is around 0.7-0.8 (some repetition but not too much)
        if 0.6 <= normalized_entropy <= 0.85:
            balance = 1.0
        else:
            balance = 1.0 - abs(normalized_entropy - 0.7) / 0.3
        
        return max(0.0, balance)


class ColorMusicAlignmentMetrics:
    """Evaluates alignment between image colors and generated music."""
    
    def __init__(self):
        """Initialize color-music alignment evaluator."""
        self.logger = logging.getLogger(__name__)
    
    def evaluate_alignment(
        self,
        colors: List[Tuple[int, int, int]],
        wavelengths: List[float],
        frequencies: List[float],
        melody: List[int]
    ) -> Dict[str, float]:
        """
        Evaluate alignment between colors and music.
        
        Args:
            colors: RGB color tuples
            wavelengths: Corresponding wavelengths
            frequencies: Musical frequencies
            melody: Generated melody notes
            
        Returns:
            Dictionary with alignment metrics
        """
        
        metrics = {}
        
        # Color preservation in melody
        metrics['color_preservation'] = self._evaluate_color_preservation(
            colors, melody
        )
        
        # Wavelength-frequency mapping consistency
        metrics['mapping_consistency'] = self._evaluate_mapping_consistency(
            wavelengths, frequencies
        )
        
        # Color harmony vs musical harmony
        metrics['harmony_alignment'] = self._evaluate_harmony_alignment(
            colors, melody
        )
        
        # Brightness-volume correlation
        metrics['brightness_energy'] = self._evaluate_brightness_energy(
            colors, melody
        )
        
        # Overall alignment
        metrics['overall_alignment'] = np.mean([
            metrics['color_preservation'],
            metrics['mapping_consistency'],
            metrics['harmony_alignment']
        ])
        
        return metrics
    
    def _evaluate_color_preservation(
        self, 
        colors: List[Tuple[int, int, int]], 
        melody: List[int]
    ) -> float:
        """Evaluate how well original colors are preserved in melody."""
        
        if not colors or not melody:
            return 0.0
        
        # Simple measure: variety in melody should reflect variety in colors
        color_variety = len(set(colors)) / len(colors) if colors else 0
        melody_variety = len(set(melody)) / len(melody) if melody else 0
        
        # Alignment is better when varieties are similar
        variety_alignment = 1.0 - abs(color_variety - melody_variety)
        
        return max(0.0, variety_alignment)
    
    def _evaluate_mapping_consistency(
        self, 
        wavelengths: List[float], 
        frequencies: List[float]
    ) -> float:
        """Evaluate consistency of wavelength-to-frequency mapping."""
        
        if len(wavelengths) != len(frequencies) or not wavelengths:
            return 0.0
        
        # Check if mapping preserves relative ordering
        wavelength_order = np.argsort(wavelengths)
        frequency_order = np.argsort(frequencies)
        
        # Calculate Spearman correlation (rank correlation)
        order_correlation = self._spearman_correlation(wavelength_order, frequency_order)
        
        return max(0.0, order_correlation)
    
    def _evaluate_harmony_alignment(
        self, 
        colors: List[Tuple[int, int, int]], 
        melody: List[int]
    ) -> float:
        """Evaluate alignment between color harmony and musical harmony."""
        
        if not colors or not melody:
            return 0.5
        
        # Calculate color harmony (based on hue relationships)
        color_harmony = self._calculate_color_harmony(colors)
        
        # Calculate musical harmony (based on interval relationships)
        musical_harmony = self._calculate_musical_harmony(melody)
        
        # Alignment is better when both are high or both are low
        harmony_diff = abs(color_harmony - musical_harmony)
        alignment = 1.0 - harmony_diff
        
        return max(0.0, alignment)
    
    def _evaluate_brightness_energy(
        self, 
        colors: List[Tuple[int, int, int]], 
        melody: List[int]
    ) -> float:
        """Evaluate alignment between color brightness and musical energy."""
        
        if not colors or not melody:
            return 0.5
        
        # Calculate average brightness
        brightnesses = [np.mean([r, g, b]) for r, g, b in colors]
        avg_brightness = np.mean(brightnesses) / 255.0
        
        # Calculate musical energy (based on pitch range and movement)
        pitch_range = max(melody) - min(melody) if melody else 0
        intervals = [abs(melody[i+1] - melody[i]) for i in range(len(melody)-1)]
        avg_movement = np.mean(intervals) if intervals else 0
        
        musical_energy = min(1.0, (pitch_range / 12.0 + avg_movement / 5.0) / 2.0)
        
        # Alignment is better when bright colors correspond to energetic music
        energy_alignment = 1.0 - abs(avg_brightness - musical_energy)
        
        return max(0.0, energy_alignment)
    
    def _calculate_color_harmony(self, colors: List[Tuple[int, int, int]]) -> float:
        """Calculate color harmony score."""
        
        if not colors:
            return 0.5
        
        # Convert to HSV and analyze hue relationships
        hues = []
        for r, g, b in colors:
            # Simple RGB to HSV conversion for hue
            max_val = max(r, g, b)
            min_val = min(r, g, b)
            diff = max_val - min_val
            
            if diff == 0:
                hue = 0
            elif max_val == r:
                hue = (60 * ((g - b) / diff) + 360) % 360
            elif max_val == g:
                hue = (60 * ((b - r) / diff) + 120) % 360
            else:
                hue = (60 * ((r - g) / diff) + 240) % 360
            
            hues.append(hue)
        
        # Harmony is higher when hues are related (complementary, triadic, etc.)
        if len(hues) < 2:
            return 0.5
        
        # Calculate hue differences
        hue_diffs = []
        for i in range(len(hues)):
            for j in range(i+1, len(hues)):
                diff = min(abs(hues[i] - hues[j]), 360 - abs(hues[i] - hues[j]))
                hue_diffs.append(diff)
        
        # Harmonious relationships: 60°, 120°, 180° (complementary)
        harmonious_angles = [60, 120, 180]
        harmony_score = 0.0
        
        for diff in hue_diffs:
            closest_harmonic = min(harmonious_angles, key=lambda x: abs(x - diff))
            closeness = 1.0 - abs(closest_harmonic - diff) / 60.0
            harmony_score += max(0.0, closeness)
        
        return min(1.0, harmony_score / len(hue_diffs))
    
    def _calculate_musical_harmony(self, melody: List[int]) -> float:
        """Calculate musical harmony score based on intervals."""
        
        if len(melody) < 2:
            return 0.5
        
        # Calculate intervals
        intervals = [abs(melody[i+1] - melody[i]) for i in range(len(melody)-1)]
        
        # Harmonious intervals in semitones: unison(0), 3rd(2), 4th(3), 5th(4), octave(7)
        harmonious_intervals = [0, 2, 3, 4, 7]
        
        harmony_score = 0.0
        for interval in intervals:
            closest_harmonic = min(harmonious_intervals, key=lambda x: abs(x - interval))
            closeness = 1.0 - abs(closest_harmonic - interval) / 6.0  # Max distance is 6 semitones
            harmony_score += max(0.0, closeness)
        
        return harmony_score / len(intervals)
    
    def _spearman_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Spearman rank correlation coefficient."""
        
        if len(x) != len(y) or len(x) == 0:
            return 0.0
        
        # Calculate ranks
        x_ranks = np.argsort(np.argsort(x))
        y_ranks = np.argsort(np.argsort(y))
        
        # Calculate correlation
        n = len(x)
        d_squared = np.sum((x_ranks - y_ranks) ** 2)
        
        if n == 1:
            return 1.0
        
        correlation = 1.0 - (6 * d_squared) / (n * (n**2 - 1))
        return correlation


class SystemPerformanceMetrics:
    """Evaluates overall system performance and efficiency."""
    
    def __init__(self):
        """Initialize system performance evaluator."""
        self.logger = logging.getLogger(__name__)
    
    def evaluate_system(
        self,
        processing_times: Dict[str, float],
        output_quality: Dict[str, float],
        user_feedback: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Evaluate overall system performance.
        
        Args:
            processing_times: Times for each pipeline component
            output_quality: Quality metrics for output
            user_feedback: Optional user feedback scores
            
        Returns:
            Dictionary with system metrics
        """
        
        metrics = {}
        
        # Processing efficiency
        metrics['total_processing_time'] = sum(processing_times.values())
        metrics['efficiency_score'] = self._calculate_efficiency_score(processing_times)
        
        # Output quality
        metrics['output_quality'] = np.mean(list(output_quality.values()))
        
        # Component balance
        metrics['component_balance'] = self._evaluate_component_balance(processing_times)
        
        # User satisfaction (if available)
        if user_feedback:
            metrics['user_satisfaction'] = np.mean(list(user_feedback.values()))
        
        # Overall system score
        scores_to_average = [
            metrics['efficiency_score'],
            metrics['output_quality'],
            metrics['component_balance']
        ]
        
        if user_feedback:
            scores_to_average.append(metrics['user_satisfaction'])
        
        metrics['overall_score'] = np.mean(scores_to_average)
        
        return metrics
    
    def _calculate_efficiency_score(self, processing_times: Dict[str, float]) -> float:
        """Calculate efficiency score based on processing times."""
        
        total_time = sum(processing_times.values())
        
        # Efficiency thresholds (in seconds)
        excellent_time = 5.0
        good_time = 15.0
        acceptable_time = 30.0
        
        if total_time <= excellent_time:
            return 1.0
        elif total_time <= good_time:
            return 0.8 + 0.2 * (good_time - total_time) / (good_time - excellent_time)
        elif total_time <= acceptable_time:
            return 0.5 + 0.3 * (acceptable_time - total_time) / (acceptable_time - good_time)
        else:
            return max(0.1, 0.5 * (60.0 - total_time) / 30.0)  # Up to 60s
    
    def _evaluate_component_balance(self, processing_times: Dict[str, float]) -> float:
        """Evaluate balance between component processing times."""
        
        if not processing_times:
            return 0.0
        
        times = list(processing_times.values())
        total_time = sum(times)
        
        if total_time == 0:
            return 1.0
        
        # Calculate coefficient of variation
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        if mean_time == 0:
            return 1.0
        
        cv = std_time / mean_time
        
        # Good balance has low coefficient of variation
        # CV < 0.5 is good, CV > 1.5 is poor
        if cv <= 0.5:
            balance_score = 1.0
        elif cv <= 1.5:
            balance_score = 1.0 - (cv - 0.5) / 1.0
        else:
            balance_score = max(0.1, 1.0 - cv / 2.0)
        
        return balance_score


class ComprehensiveEvaluator:
    """Comprehensive evaluation combining all metrics."""
    
    def __init__(self):
        """Initialize comprehensive evaluator."""
        self.logger = logging.getLogger(__name__)
        
        self.musical_metrics = MusicalQualityMetrics()
        self.alignment_metrics = ColorMusicAlignmentMetrics()
        self.system_metrics = SystemPerformanceMetrics()
    
    def evaluate_complete_pipeline(
        self,
        colors: List[Tuple[int, int, int]],
        wavelengths: List[float],
        frequencies: List[float],
        melody: List[int],
        scale: List[int],
        processing_times: Dict[str, float],
        user_feedback: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of the entire pipeline.
        
        Args:
            colors: Extracted colors
            wavelengths: Converted wavelengths
            frequencies: Musical frequencies
            melody: Generated melody
            scale: Musical scale used
            processing_times: Component processing times
            user_feedback: Optional user feedback
            
        Returns:
            Complete evaluation report
        """
        
        evaluation = {}
        
        # Musical quality evaluation
        evaluation['musical_quality'] = self.musical_metrics.evaluate_melody(melody, scale)
        
        # Color-music alignment evaluation
        evaluation['alignment_quality'] = self.alignment_metrics.evaluate_alignment(
            colors, wavelengths, frequencies, melody
        )
        
        # System performance evaluation
        evaluation['system_performance'] = self.system_metrics.evaluate_system(
            processing_times, 
            {**evaluation['musical_quality'], **evaluation['alignment_quality']},
            user_feedback
        )
        
        # Overall evaluation
        evaluation['overall'] = {
            'musical_score': evaluation['musical_quality']['overall_quality'],
            'alignment_score': evaluation['alignment_quality']['overall_alignment'],
            'system_score': evaluation['system_performance']['overall_score'],
            'final_score': np.mean([
                evaluation['musical_quality']['overall_quality'],
                evaluation['alignment_quality']['overall_alignment'],
                evaluation['system_performance']['overall_score']
            ])
        }
        
        # Add recommendations
        evaluation['recommendations'] = self._generate_recommendations(evaluation)
        
        self.logger.info(f"Comprehensive evaluation completed. Final score: {evaluation['overall']['final_score']:.2f}")
        
        return evaluation
    
    def _generate_recommendations(self, evaluation: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improvement based on evaluation results."""
        
        recommendations = []
        
        # Musical quality recommendations
        musical = evaluation['musical_quality']
        if musical['contour_smoothness'] < 0.6:
            recommendations.append("Consider using smoother melodic motion with smaller intervals")
        
        if musical['interval_variety'] < 0.5:
            recommendations.append("Increase melodic interval variety for more interesting melodies")
        
        if musical['phrase_structure'] < 0.6:
            recommendations.append("Improve phrase structure with clearer musical sentences")
        
        # Alignment recommendations
        alignment = evaluation['alignment_quality']
        if alignment['color_preservation'] < 0.5:
            recommendations.append("Better preserve original color characteristics in the melody")
        
        if alignment['harmony_alignment'] < 0.6:
            recommendations.append("Improve alignment between color harmony and musical harmony")
        
        # System performance recommendations
        system = evaluation['system_performance']
        if system['efficiency_score'] < 0.7:
            recommendations.append("Optimize processing pipeline for better performance")
        
        if system.get('component_balance', 1.0) < 0.6:
            recommendations.append("Balance processing times across pipeline components")
        
        # Overall recommendations
        overall_score = evaluation['overall']['final_score']
        if overall_score < 0.5:
            recommendations.append("Consider reviewing the entire pipeline architecture")
        elif overall_score < 0.7:
            recommendations.append("Focus on the lowest-scoring evaluation categories")
        
        return recommendations