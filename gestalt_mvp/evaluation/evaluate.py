#!/usr/bin/env python3
"""
Gestalt-MVP Evaluation Script

Runs locked evaluation on Sessions 4-5 after tuning thresholds on Session 3.
Generates confusion matrix, FAR, FRR, macro F1, and bootstrap confidence intervals.

Usage:
    python evaluate.py --data-dir ../data_collection/data --results-dir results
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
    
    Args:
        frames_a: List of frames, each frame is 21×[x,y,z]
        frames_b: List of frames, each frame is 21×[x,y,z]
    
    Returns:
        DTW accumulated distance
    """
    n = len(frames_a)
    m = len(frames_b)
    
    if n == 0 or m == 0:
        return float('inf')
    
    band = sakoe_chiba_band(n, m)
    
    # Initialize DP matrix
    dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    
    # Fill DP matrix with band constraint
    for i in range(1, n + 1):
        j_start = max(1, i - band)
        j_end = min(m, i + band)
        
        for j in range(j_start, j_end + 1):
            cost = weighted_euclidean_distance(frames_a[i-1], frames_b[j-1])
            
            dp[i][j] = cost + min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])
    
    return dp[n][m]


class GestureMatcher:
    """DTW-based gesture matcher with multi-exemplar support."""
    
    def __init__(self):
        self.exemplars: Dict[str, List[List[List[List[float]]]]] = defaultdict(list)
    
    def enroll(self, gesture_id: str, trajectory_frames: List[List[List[float]]]):
        """Enroll an exemplar for a gesture class."""
        self.exemplars[gesture_id].append(trajectory_frames)
    
    def match(self, query_frames: List[List[List[float]]]) -> Tuple[Optional[str], float, float]:
        """
        Match query against all enrolled exemplars.
        
        Returns:
            (best_gesture_id, D1, D2) where D1=best distance, D2=second-best
        """
        if not self.exemplars:
            return None, float('inf'), float('inf')
        
        distances = []
        
        for gesture_id, exemplars in self.exemplars.items():
            min_dist = min(dtw_distance(query_frames, ex) for ex in exemplars)
            distances.append((gesture_id, min_dist))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        best_gesture = distances[0][0]
        d1 = distances[0][1]
        d2 = distances[1][1] if len(distances) > 1 else float('inf')
        
        return best_gesture, d1, d2


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
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        per_class_metrics[gesture] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': confusion[i, :].sum()
        }
    
    # Macro F1
    macro_f1 = np.mean([m['f1'] for m in per_class_metrics.values()])
    
    # Cost function: Cost = 10 × FP + 2 × FR + 0.5 × Clarifications
    fp = sum(confusion[:, i].sum() - confusion[i, i] for i in range(10))
    fr = rejected_known
    cost = 10 * fp + 2 * fr + 0.5 * clarifications
    
    return {
        'confusion_matrix': confusion.tolist(),
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'far': far,
        'frr': frr,
        'near_miss_rejection_rate': near_miss_rate,
        'per_class_metrics': per_class_metrics,
        'cost': cost,
        'total_known': total_known,
        'total_unknown': total_unknown,
        'clarifications': clarifications,
    }


def bootstrap_confidence_intervals(data_dir: Path,
                                    enrollment_sessions: List[int],
                                    calibration_session: int,
                                    test_sessions: List[int],
                                    tau_accept: float,
                                    tau_margin: float,
                                    n_bootstrap: int = 1000) -> Dict[str, Tuple[float, float]]:
    """
    Compute session-level bootstrap confidence intervals.
    
    Returns 95% CI for key metrics.
    """
    # Load all test data
    test_segments = []
    for session in test_sessions:
        test_segments.extend(load_session_data(data_dir, session))
    
    if len(test_segments) == 0:
        return {}
    
    # Bootstrap resampling at session level
    bootstrap_metrics = {
        'macro_f1': [],
        'far': [],
        'frr': [],
        'accuracy': [],
    }
    
    sessions_in_test = list(set(s for s in test_sessions))
    
    for _ in range(n_bootstrap):
        # Resample sessions with replacement
        resampled_sessions = np.random.choice(sessions_in_test, 
                                               size=len(sessions_in_test),
                                               replace=True)
        
        resampled_segments = []
        for session in resampled_sessions:
            session_data = load_session_data(data_dir, int(session))
            resampled_segments.extend(session_data)
        
        if len(resampled_segments) == 0:
            continue
        
        # Build matcher from enrollment data
        matcher = GestureMatcher()
        for session in enrollment_sessions:
            segments = load_session_data(data_dir, session)
            for seg in segments:
                frames = [f['landmarks'] for f in seg['frames']]
                matcher.enroll(seg['gesture_id'], frames)
        
        # Evaluate
        metrics = evaluate_with_thresholds(
            matcher, resampled_segments, tau_accept, tau_margin
        )
        
        bootstrap_metrics['macro_f1'].append(metrics['macro_f1'])
        bootstrap_metrics['far'].append(metrics['far'])
        bootstrap_metrics['frr'].append(metrics['frr'])
        bootstrap_metrics['accuracy'].append(metrics['accuracy'])
    
    # Calculate 95% CIs
    cis = {}
    for metric, values in bootstrap_metrics.items():
        if len(values) > 0:
            lower = np.percentile(values, 2.5)
            upper = np.percentile(values, 97.5)
            cis[metric] = (float(lower), float(upper))
    
    return cis


def tune_thresholds(data_dir: Path,
                    enrollment_sessions: List[int],
                    calibration_session: int,
                    target_far: float = 0.05) -> Tuple[float, float]:
    """
    Tune acceptance and margin thresholds on calibration session.
    
    Goal: Achieve target FAR while maximizing F1.
    """
    # Build matcher from enrollment data
    matcher = GestureMatcher()
    for session in enrollment_sessions:
        segments = load_session_data(data_dir, session)
        for seg in segments:
            frames = [f['landmarks'] for f in seg['frames']]
            matcher.enroll(seg['gesture_id'], frames)
    
    # Load calibration data
    cal_segments = load_session_data(data_dir, calibration_session)
    
    if len(cal_segments) == 0:
        logger.warning("No calibration data available, using defaults")
        return 0.5, 0.15
    
    # Grid search over thresholds
    tau_accept_values = np.linspace(0.2, 0.8, 20)
    tau_margin_values = np.linspace(0.05, 0.3, 15)
    
    best_tau_accept = 0.5
    best_tau_margin = 0.15
    best_score = -float('inf')
    
    for tau_acc in tau_accept_values:
        for tau_mar in tau_margin_values:
            metrics = evaluate_with_thresholds(matcher, cal_segments, tau_acc, tau_mar)
            
            # Score: maximize F1 while penalizing FAR > target
            f1 = metrics['macro_f1']
            far = metrics['far']
            
            if far <= target_far:
                score = f1
            else:
                score = f1 - 2.0 * (far - target_far)
            
            if score > best_score:
                best_score = score
                best_tau_accept = tau_acc
                best_tau_margin = tau_mar
    
    logger.info(f"Tuned thresholds: tau_accept={best_tau_accept:.3f}, "
               f"tau_margin={best_tau_margin:.3f}")
    
    return best_tau_accept, best_tau_margin


def main():
    parser = argparse.ArgumentParser(description='Gestalt-MVP Evaluation Script')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing session data')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory for output results')
    parser.add_argument('--target-far', type=float, default=0.05,
                       help='Target false acceptance rate')
    parser.add_argument('--n-bootstrap', type=int, default=1000,
                       help='Number of bootstrap iterations')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define session splits (frozen from Section 9)
    enrollment_sessions = [1, 2]
    calibration_session = 3
    test_sessions = [4, 5]
    
    # Step 1: Tune thresholds on Session 3
    logger.info("Tuning thresholds on calibration session (Session 3)...")
    tau_accept, tau_margin = tune_thresholds(
        data_dir, enrollment_sessions, calibration_session, args.target_far
    )
    
    logger.info(f"FROZEN thresholds: tau_accept={tau_accept:.4f}, tau_margin={tau_margin:.4f}")
    
    # Step 2: Build matcher from enrollment data
    logger.info("Building matcher from enrollment sessions (Sessions 1-2)...")
    matcher = GestureMatcher()
    
    for session in enrollment_sessions:
        segments = load_session_data(data_dir, session)
        for seg in segments:
            frames = [f['landmarks'] for f in seg['frames']]
            matcher.enroll(seg['gesture_id'], frames)
    
    logger.info(f"Enrolled {sum(len(v) for v in matcher.exemplars.values())} exemplars")
    
    # Step 3: Run locked evaluation on Sessions 4-5
    logger.info("Running locked evaluation on test sessions (Sessions 4-5)...")
    test_segments = []
    for session in test_sessions:
        test_segments.extend(load_session_data(data_dir, session))
    
    if len(test_segments) == 0:
        logger.error("No test data found!")
        sys.exit(1)
    
    metrics = evaluate_with_thresholds(
        matcher, test_segments, tau_accept, tau_margin
    )
    
    # Step 4: Compute bootstrap confidence intervals
    logger.info(f"Computing bootstrap confidence intervals ({args.n_bootstrap} iterations)...")
    cis = bootstrap_confidence_intervals(
        data_dir, enrollment_sessions, calibration_session, test_sessions,
        tau_accept, tau_margin, args.n_bootstrap
    )
    
    # Step 5: Save results
    timestamp = datetime.now().isoformat()
    results = {
        'timestamp': timestamp,
        'thresholds': {
            'tau_accept': tau_accept,
            'tau_margin': tau_margin,
        },
        'primary_endpoint': {
            'macro_f1': metrics['macro_f1'],
            'far': metrics['far'],
            'passed': metrics['macro_f1'] > 0.85 and metrics['far'] < 0.05,
        },
        'metrics': metrics,
        'bootstrap_cis': cis,
        'enrollment_count': sum(len(v) for v in matcher.exemplars.values()),
        'test_samples': len(test_segments),
    }
    
    # Save JSON
    results_file = results_dir / f"evaluation_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV summary
    csv_file = results_dir / f"summary_{timestamp}.csv"
    with open(csv_file, 'w') as f:
        f.write("Metric,Value\n")
        f.write(f"Macro F1,{metrics['macro_f1']:.4f}\n")
        f.write(f"FAR,{metrics['far']:.4f}\n")
        f.write(f"FRR,{metrics['frr']:.4f}\n")
        f.write(f"Accuracy,{metrics['accuracy']:.4f}\n")
        f.write(f"Near-miss Rejection,{metrics['near_miss_rejection_rate']:.4f}\n")
        f.write(f"Cost,{metrics['cost']:.2f}\n")
        f.write(f"Clarifications,{metrics['clarifications']}\n")
        
        if cis:
            f.write(f"\n")
            f.write(f"Metric,Lower CI,Upper CI\n")
            for metric, (lower, upper) in cis.items():
                f.write(f"{metric},{lower:.4f},{upper:.4f}\n")
    
    # Print summary
    print("\n" + "="*60)
    print("GESTALT-MVP EVALUATION RESULTS")
    print("="*60)
    print(f"Thresholds: tau_accept={tau_accept:.4f}, tau_margin={tau_margin:.4f}")
    print(f"Test samples: {len(test_segments)}")
    print(f"\nPRIMARY ENDPOINT:")
    print(f"  Macro F1: {metrics['macro_f1']:.4f} {'✓' if metrics['macro_f1'] > 0.85 else '✗'}")
    print(f"  FAR:      {metrics['far']:.4f} {'✓' if metrics['far'] < 0.05 else '✗'}")
    print(f"  PASSED:   {'YES' if results['primary_endpoint']['passed'] else 'NO'}")
    print(f"\nSECONDARY METRICS:")
    print(f"  FRR:     {metrics['frr']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Near-miss rejection: {metrics['near_miss_rejection_rate']:.4f}")
    print(f"  Cost:    {metrics['cost']:.2f}")
    
    if cis:
        print(f"\nBOOTSTRAP 95% CONFIDENCE INTERVALS:")
        for metric, (lower, upper) in cis.items():
            print(f"  {metric}: [{lower:.4f}, {upper:.4f}]")
    
    print(f"\nResults saved to: {results_file}")
    print("="*60)
    
    # Exit with appropriate code
    sys.exit(0 if results['primary_endpoint']['passed'] else 1)


if __name__ == '__main__':
    main()
