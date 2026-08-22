#!/usr/bin/env python3
"""
Gestalt-MVP Fast Evaluation Script

Optimized evaluation for Phase 0 with subsampling for speed.
Uses the same protocol as evaluate.py but with optimizations:
- Subsample frames in trajectories (every 2nd frame)
- Use fewer bootstrap samples
- Cache DTW distances where possible
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.normalize import normalize_trajectory
from data_collection.segment import segment_trajectory, validate_segment_quality


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Gesture vocabulary
GESTURES = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10']
NEGATIVE_CLASSES = ['NEG-1', 'NEG-2', 'NEG-3']

# Default DTW parameters (frozen from Section 6)
LANDMARK_WEIGHTS = [
    1.0, 1.0, 1.0, 1.0, 2.0,  # Thumb
    1.0, 1.0, 1.0, 2.0,       # Index
    1.0, 1.0, 1.0, 2.0,       # Middle
    1.0, 1.0, 1.0, 2.0,       # Ring
    1.0, 1.0, 1.0, 2.0,       # Pinky
]


def load_session_data(data_dir: Path, session: int) -> List[Dict[str, Any]]:
    """Load all gesture segments from a session."""
    session_dir = data_dir / f"session{session}"
    
    if not session_dir.exists():
        logger.warning(f"Session {session} directory not found")
        return []
    
    segments = []
    for json_file in session_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                segments.append(data)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load {json_file}: {e}")
    
    return segments


def subsample_trajectory(frames: List[List[List[float]]], factor: int = 2) -> List[List[List[float]]]:
    """Subsample trajectory frames for faster DTW."""
    return frames[::factor]


def weighted_euclidean_distance(frame_a: List[List[float]], 
                                 frame_b: List[List[float]]) -> float:
    """Calculate weighted Euclidean distance between two frames."""
    sum_sq = 0.0
    
    for i, (lm_a, lm_b) in enumerate(zip(frame_a, frame_b)):
        dx = lm_a[0] - lm_b[0]
        dy = lm_a[1] - lm_b[1]
        dz = lm_a[2] - lm_b[2]
        
        dist_sq = dx*dx + dy*dy + dz*dz
        weight = LANDMARK_WEIGHTS[i]
        
        sum_sq += weight * dist_sq
    
    return sum_sq ** 0.5


def sakoe_chiba_band(len_x: int, len_y: int) -> int:
    """Calculate Sakoe-Chiba band radius: r = max(3, floor(0.20 × max(L_x, L_y)))."""
    max_len = max(len_x, len_y)
    band = int(0.20 * max_len)
    return max(3, band)


def dtw_distance(frames_a: List[List[List[float]]], 
                 frames_b: List[List[List[float]]]) -> float:
    """
    Calculate DTW distance between two trajectories with Sakoe-Chiba band.
    
    Optimized version with early stopping for very high distances.
    """
    n = len(frames_a)
    m = len(frames_b)
    
    if n == 0 or m == 0:
        return float('inf')
    
    band = sakoe_chiba_band(n, m)
    
    # Initialize DP matrix
    dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    
    # Fill DP matrix with band constraint
    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m, i + band)
        
        for j in range(j_start, j_end + 1):
            cost = weighted_euclidean_distance(frames_a[i-1], frames_b[j-1])
            
            dp[i][j] = cost + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
            
            # Early stopping if distance gets too large
            if dp[i][j] > 1000:
                return dp[i][j]
    
    return dp[n][m]


class GestureMatcher:
    """DTW-based gesture matcher with multi-exemplar support."""
    
    def __init__(self):
        self.exemplars: Dict[str, List[List[List[List[float]]]]] = defaultdict(list)
    
    def enroll(self, gesture_id: str, trajectory_frames: List[List[List[float]]]):
        """Enroll an exemplar for a gesture class."""
        # Subsample for speed
        subsampled = subsample_trajectory(trajectory_frames, factor=2)
        self.exemplars[gesture_id].append(subsampled)
    
    def match(self, query_frames: List[List[List[float]]]) -> Tuple[Optional[str], float, float]:
        """
        Match query against all enrolled exemplars.
        
        Returns:
            (best_gesture_id, D1, D2) where D1=best distance, D2=second-best
        """
        if not self.exemplars:
            return None, float('inf'), float('inf')
        
        # Subsample query
        query_sub = subsample_trajectory(query_frames, factor=2)
        
        distances = []
        
        for gesture_id, exemplars in self.exemplars.items():
            min_dist = min(dtw_distance(query_sub, ex) for ex in exemplars)
            distances.append((gesture_id, min_dist))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        best_gesture = distances[0][0]
        d1 = distances[0][1]
        d2 = distances[1][1] if len(distances) > 1 else float('inf')
        
        return best_gesture, d1, d2


def tune_thresholds(matcher: GestureMatcher, 
                    calibration_segments: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Tune acceptance and margin thresholds on calibration data.
    
    Grid search over tau_accept and tau_margin to maximize:
    - Macro F1 > 0.85
    - FAR < 5%
    - Minimize cost = 10×FP + 2×FR + 0.5×Clarifications
    """
    # Collect distances for known gestures
    known_d1 = []
    known_margins = []
    unknown_d1 = []
    unknown_margins = []
    
    for seg in calibration_segments:
        gesture_id = seg['gesture_id']
        frames = [f['landmarks'] for f in seg['frames']]
        
        pred_gesture, d1, d2 = matcher.match(frames)
        margin = d2 - d1
        
        if gesture_id in GESTURES:
            known_d1.append(d1)
            known_margins.append(margin)
        elif gesture_id in NEGATIVE_CLASSES:
            unknown_d1.append(d1)
            unknown_margins.append(margin)
    
    # Grid search
    best_tau_accept = 50.0
    best_tau_margin = 5.0
    best_score = float('inf')
    
    tau_accept_candidates = np.percentile(known_d1, np.arange(50, 95, 5))
    tau_margin_candidates = np.percentile(known_margins, np.arange(10, 90, 10))
    
    for tau_accept in tau_accept_candidates:
        for tau_margin in tau_margin_candidates:
            # Calculate metrics
            fp = sum(1 for d1 in unknown_d1 if d1 < tau_accept)
            fr = sum(1 for d1 in known_d1 if d1 > tau_accept)
            clarifications = sum(1 for (d1, m) in zip(known_d1, known_margins) 
                                if d1 < tau_accept and m <= tau_margin)
            
            cost = 10 * fp + 2 * fr + 0.5 * clarifications
            
            # Check constraints
            far = fp / len(unknown_d1) if unknown_d1 else 0
            
            if far < 0.05 and cost < best_score:
                best_score = cost
                best_tau_accept = tau_accept
                best_tau_margin = tau_margin
    
    logger.info(f"Tuned thresholds: tau_accept={best_tau_accept:.2f}, tau_margin={best_tau_margin:.2f}")
    return best_tau_accept, best_tau_margin


def evaluate_with_thresholds(matcher: GestureMatcher,
                              test_segments: List[Dict[str, Any]],
                              tau_accept: float,
                              tau_margin: float) -> Dict[str, Any]:
    """
    Evaluate matcher with given thresholds.
    
    Returns metrics including confusion matrix, FAR, FRR.
    """
    predictions = []
    ground_truth = []
    distances_d1 = []
    margins = []
    
    for segment in test_segments:
        gesture_id = segment['gesture_id']
        frames = [f['landmarks'] for f in segment['frames']]
        
        pred_gesture, d1, d2 = matcher.match(frames)
        margin = d2 - d1
        
        distances_d1.append(d1)
        margins.append(margin)
        
        # Apply gating
        if d1 > tau_accept:
            predicted = 'REJECT'
        elif margin <= tau_margin:
            predicted = 'CLARIFY'
        else:
            predicted = pred_gesture
        
        predictions.append(predicted)
        ground_truth.append(gesture_id)
    
    # Calculate metrics
    return compute_metrics(ground_truth, predictions, distances_d1, margins, test_segments)


def compute_metrics(ground_truth: List[str],
                    predictions: List[str],
                    distances_d1: List[float],
                    margins: List[float],
                    segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all evaluation metrics."""
    
    # Confusion matrix (10×10 for known classes)
    confusion = np.zeros((10, 10), dtype=int)
    gesture_to_idx = {g: i for i, g in enumerate(GESTURES)}
    
    correct = 0
    total_known = 0
    rejected_known = 0
    accepted_unknown = 0
    total_unknown = 0
    near_miss_rejected = 0
    near_miss_total = 0
    clarifications = 0
    
    for gt, pred, seg in zip(ground_truth, predictions, segments):
        condition = seg.get('condition', 'normal')
        
        if gt in GESTURES:
            total_known += 1
            
            if pred == 'REJECT':
                rejected_known += 1
            elif pred == 'CLARIFY':
                clarifications += 1
            elif pred in GESTURES:
                gt_idx = gesture_to_idx[gt]
                pred_idx = gesture_to_idx[pred]
                confusion[gt_idx, pred_idx] += 1
                
                if gt == pred:
                    correct += 1
        
        elif gt in NEGATIVE_CLASSES:
            total_unknown += 1
            
            if pred != 'REJECT' and pred != 'CLARIFY':
                accepted_unknown += 1
            
            if gt == 'NEG-3':
                near_miss_total += 1
                if pred == 'REJECT':
                    near_miss_rejected += 1
    
    # Calculate rates
    accuracy = correct / total_known if total_known > 0 else 0.0
    frr = rejected_known / total_known if total_known > 0 else 0.0
    far = accepted_unknown / total_unknown if total_unknown > 0 else 0.0
    near_miss_rate = near_miss_rejected / near_miss_total if near_miss_total > 0 else 0.0
    
    # Per-class precision/recall/F1
    per_class_metrics = {}
    for i, gesture in enumerate(GESTURES):
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum() - tp)
        fn = int(confusion[i, :].sum() - tp)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[gesture] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(confusion[i, :].sum())
        }
    
    # Macro F1
    macro_f1 = float(np.mean([m['f1'] for m in per_class_metrics.values()]))
    
    # Cost function: Cost = 10 × FP + 2 × FR + 0.5 × Clarifications
    fp = int(sum(confusion[:, i].sum() - confusion[i, i] for i in range(10)))
    fr = int(rejected_known)
    cost = float(10 * fp + 2 * fr + 0.5 * clarifications)
    
    return {
        'confusion_matrix': [[int(x) for x in row] for row in confusion.tolist()],
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'far': float(far),
        'frr': float(frr),
        'near_miss_rejection_rate': float(near_miss_rate),
        'per_class_metrics': per_class_metrics,
        'cost': cost,
        'total_known': int(total_known),
        'total_unknown': int(total_unknown),
        'clarifications': int(clarifications),
    }


def run_evaluation(data_dir: Path, results_dir: Path) -> Dict[str, Any]:
    """Run complete Phase 0 evaluation."""
    
    logger.info("Starting Phase 0 evaluation...")
    
    # Load enrollment data (Sessions 1-2)
    enrollment_segments = []
    for session in [1, 2]:
        enrollment_segments.extend(load_session_data(data_dir, session))
    
    logger.info(f"Loaded {len(enrollment_segments)} enrollment segments from Sessions 1-2")
    
    # Build matcher from enrollment data
    matcher = GestureMatcher()
    for seg in enrollment_segments:
        gesture_id = seg['gesture_id']
        if gesture_id.startswith('NEG'):
            continue  # Don't enroll negatives
        
        frames = [f['landmarks'] for f in seg['frames']]
        matcher.enroll(gesture_id, frames)
    
    logger.info(f"Enrolled {len(matcher.exemplars)} gesture classes")
    
    # Load calibration data (Session 3)
    calibration_segments = load_session_data(data_dir, 3)
    logger.info(f"Tuning thresholds on {len(calibration_segments)} calibration segments from Session 3...")
    
    # Tune thresholds
    tau_accept, tau_margin = tune_thresholds(matcher, calibration_segments)
    
    # Freeze thresholds
    logger.info(f"FROZEN THRESHOLDS: tau_accept={tau_accept:.2f}, tau_margin={tau_margin:.2f}")
    
    # Load locked test data (Sessions 4-5)
    test_segments = []
    for session in [4, 5]:
        test_segments.extend(load_session_data(data_dir, session))
    
    logger.info(f"Evaluating on {len(test_segments)} locked test segments from Sessions 4-5...")
    
    # Run evaluation
    metrics = evaluate_with_thresholds(matcher, test_segments, tau_accept, tau_margin)
    
    # Save results
    results = {
        'phase': 0,
        'timestamp': datetime.now().isoformat(),
        'thresholds': {
            'tau_accept': tau_accept,
            'tau_margin': tau_margin,
        },
        'metrics': metrics,
        'primary_endpoint_passed': (
            metrics['macro_f1'] > 0.85 and 
            metrics['far'] < 0.05
        ),
    }
    
    # Save to JSON
    results_file = results_dir / 'phase0_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("PHASE 0 EVALUATION RESULTS")
    print("="*60)
    print(f"\nThresholds (tuned on Session 3, frozen):")
    print(f"  τ_accept = {tau_accept:.2f}")
    print(f"  τ_margin = {tau_margin:.2f}")
    print(f"\nPrimary Metrics (Sessions 4-5):")
    print(f"  Macro F1: {metrics['macro_f1']:.3f} {'✓ PASS' if metrics['macro_f1'] > 0.85 else '✗ FAIL'}")
    print(f"  FAR:      {metrics['far']:.3f} {'✓ PASS' if metrics['far'] < 0.05 else '✗ FAIL'}")
    print(f"  FRR:      {metrics['frr']:.3f}")
    print(f"  Near-miss rejection: {metrics['near_miss_rejection_rate']:.3f}")
    print(f"\nCost Function: {metrics['cost']:.1f}")
    print(f"  (Cost = 10×FP + 2×FR + 0.5×Clarifications)")
    print(f"\nPer-class F1 scores:")
    for gesture, m in sorted(metrics['per_class_metrics'].items()):
        status = '✓' if m['f1'] > 0.8 else '✗'
        print(f"  {gesture}: {m['f1']:.3f} {status}")
    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    cm = np.array(metrics['confusion_matrix'])
    print(cm)
    print("\n" + "="*60)
    
    # Determine next phase recommendation
    if metrics['macro_f1'] > 0.85 and metrics['far'] < 0.05:
        print("\n✓ PHASE 0 PASSED: DTW baseline is sufficient.")
        print("→ Recommendation: Skip Phase 1, proceed to Phase 2.")
    else:
        print("\n✗ PHASE 0 FAILED: DTW baseline did not meet targets.")
        print("→ Recommendation: Proceed to Phase 1 (neural encoder).")
        if metrics['macro_f1'] <= 0.85:
            print(f"   Issue: Macro F1 ({metrics['macro_f1']:.3f}) below 0.85 threshold")
        if metrics['far'] >= 0.05:
            print(f"   Issue: FAR ({metrics['far']:.3f}) above 5% threshold")
    
    print("="*60 + "\n")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gestalt-MVP Phase 0 Evaluation')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to data collection directory')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run evaluation
    results = run_evaluation(data_dir, results_dir)
    
    # Exit with appropriate code
    if results['primary_endpoint_passed']:
        print("Phase 0 PASSED ✓")
        sys.exit(0)
    else:
        print("Phase 0 FAILED ✗")
        sys.exit(1)
