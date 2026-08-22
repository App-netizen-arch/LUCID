//! Confidence Module for Gestalt-MVP
//! 
//! Multi-factor confidence calculation:
//! C_final = C_visual × C_gesture × C_temporal × C_context × C_symbolic

use crate::message_bus::{MatchResult, GateDecision};
use crate::graph::Trajectory;

/// Confidence factors (all in range [0, 1])
#[derive(Debug, Clone)]
pub struct ConfidenceFactors {
    /// Visual quality from MediaPipe (landmark detection confidence)
    pub c_visual: f64,
    
    /// Gesture match quality (based on DTW distance and margin)
    pub c_gesture: f64,
    
    /// Temporal consistency (duration within expected bounds)
    pub c_temporal: f64,
    
    /// Contextual appropriateness (time of day, recent history)
    pub c_context: f64,
    
    /// Symbolic validation (guard checks pass)
    pub c_symbolic: f64,
}

impl Default for ConfidenceFactors {
    fn default() -> Self {
        Self {
            c_visual: 1.0,
            c_gesture: 1.0,
            c_temporal: 1.0,
            c_context: 1.0,
            c_symbolic: 1.0,
        }
    }
}

/// Final confidence result
#[derive(Debug, Clone)]
pub struct ConfidenceResult {
    pub c_final: f64,
    pub factors: ConfidenceFactors,
    pub decision: GateDecision,
    pub explanation: String,
}

/// Calculate visual confidence from MediaPipe landmark detection
/// 
/// Based on average detection confidence across 21 landmarks.
pub fn calculate_visual_confidence(landmark_confidences: &[f32]) -> f64 {
    if landmark_confidences.is_empty() {
        return 0.0;
    }
    
    let avg: f32 = landmark_confidences.iter().sum::<f32>() / landmark_confidences.len() as f32;
    avg as f64
}

/// Calculate gesture confidence from DTW match result
/// 
/// Uses sigmoid-like function based on distance and margin.
pub fn calculate_gesture_confidence(
    match_result: &MatchResult,
    tau_accept: f64,
    tau_reject: f64,
) -> f64 {
    // Distance component: lower is better
    let dist_score = if match_result.distance_to_best <= tau_accept {
        1.0 - (match_result.distance_to_best / tau_accept) * 0.3
    } else if match_result.distance_to_best >= tau_reject {
        0.2
    } else {
        0.7 - ((match_result.distance_to_best - tau_accept) / (tau_reject - tau_accept)) * 0.5
    };
    
    // Margin component: larger is better
    let margin_threshold = 0.15; // From Section 6
    let margin_score = if match_result.margin >= margin_threshold * 2.0 {
        1.0
    } else if match_result.margin <= 0.0 {
        0.3
    } else {
        0.3 + (match_result.margin / (margin_threshold * 2.0)) * 0.7
    };
    
    // Combine with emphasis on margin (more important for open-set)
    dist_score * 0.4 + margin_score * 0.6
}

/// Calculate temporal confidence from gesture duration
/// 
/// Expected duration: 0.3s to 2.5s (Section 5)
/// Optimal: around 1.0s
pub fn calculate_temporal_confidence(duration_s: f64) -> f64 {
    const MIN_DURATION: f64 = 0.3;
    const MAX_DURATION: f64 = 2.5;
    const OPTIMAL_DURATION: f64 = 1.0;
    
    if duration_s < MIN_DURATION || duration_s > MAX_DURATION {
        return 0.0; // Should have been filtered by segmentation
    }
    
    // Gaussian-like falloff from optimal
    let deviation = (duration_s - OPTIMAL_DURATION).abs();
    let normalized_deviation = deviation / (MAX_DURATION - MIN_DURATION) * 2.0;
    
    (1.0 - normalized_deviation.powi(2)).max(0.5)
}

/// Calculate context confidence based on recent history
/// 
/// Penalizes rapid repeated gestures (likely noise or stuttering).
pub fn calculate_context_confidence(
    last_gesture_time_ms: Option<i64>,
    current_time_ms: i64,
    min_interval_ms: i64,
) -> f64 {
    match last_gesture_time_ms {
        Some(last_time) => {
            let interval = current_time_ms - last_time;
            if interval < min_interval_ms {
                // Too soon after last gesture
                (interval as f64 / min_interval_ms as f64).min(1.0)
            } else {
                1.0
            }
        }
        None => 1.0, // No previous gesture
    }
}

/// Calculate symbolic confidence from guard checks
/// 
/// All guards must pass for full confidence.
pub fn calculate_symbolic_confidence(guard_results: &[bool]) -> f64 {
    if guard_results.is_empty() {
        return 1.0;
    }
    
    let pass_count = guard_results.iter().filter(|&&r| r).count();
    pass_count as f64 / guard_results.len() as f64
}

/// Compute final multi-factor confidence
/// 
/// C_final = C_visual × C_gesture × C_temporal × C_context × C_symbolic
pub fn compute_final_confidence(factors: &ConfidenceFactors) -> f64 {
    factors.c_visual 
        * factors.c_gesture 
        * factors.c_temporal 
        * factors.c_context 
        * factors.c_symbolic
}

/// Full confidence evaluation pipeline
pub fn evaluate_confidence(
    match_result: &MatchResult,
    trajectory: &Trajectory,
    landmark_confidences: &[f32],
    last_gesture_time_ms: Option<i64>,
    current_time_ms: i64,
    guard_results: &[bool],
    tau_accept: f64,
    tau_reject: f64,
) -> ConfidenceResult {
    let factors = ConfidenceFactors {
        c_visual: calculate_visual_confidence(landmark_confidences),
        c_gesture: calculate_gesture_confidence(match_result, tau_accept, tau_reject),
        c_temporal: calculate_temporal_confidence(trajectory.duration_s),
        c_context: calculate_context_confidence(last_gesture_time_ms, current_time_ms, 500),
        c_symbolic: calculate_symbolic_confidence(guard_results),
    };
    
    let c_final = compute_final_confidence(&factors);
    
    // Determine decision based on final confidence
    let (decision, explanation) = if c_final >= 0.7 {
        (GateDecision::Accept, "High confidence acceptance".to_string())
    } else if c_final >= 0.4 {
        (GateDecision::Clarify, "Moderate confidence, clarification needed".to_string())
    } else {
        (GateDecision::Reject, "Low confidence rejection".to_string())
    };
    
    ConfidenceResult {
        c_final,
        factors,
        decision,
        explanation,
    }
}

/// Threshold calibration helper
/// 
/// Analyzes match results to suggest optimal thresholds.
pub struct ThresholdCalibrator {
    known_distances: Vec<f64>,
    unknown_distances: Vec<f64>,
    known_margins: Vec<f64>,
    unknown_margins: Vec<f64>,
}

impl ThresholdCalibrator {
    pub fn new() -> Self {
        Self {
            known_distances: Vec::new(),
            unknown_distances: Vec::new(),
            known_margins: Vec::new(),
            unknown_margins: Vec::new(),
        }
    }
    
    pub fn add_known_sample(&mut self, distance: f64, margin: f64) {
        self.known_distances.push(distance);
        self.known_margins.push(margin);
    }
    
    pub fn add_unknown_sample(&mut self, distance: f64, margin: f64) {
        self.unknown_distances.push(distance);
        self.unknown_margins.push(margin);
    }
    
    /// Calibrate tau_accept to achieve target FAR
    pub fn calibrate_tau_accept(&self, target_far: f64) -> f64 {
        if self.unknown_distances.is_empty() {
            return 0.5; // Default
        }
        
        let mut sorted = self.unknown_distances.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        // Find threshold that achieves target FAR
        let idx = ((1.0 - target_far) * sorted.len() as f64) as usize;
        sorted.get(idx).copied().unwrap_or(0.5)
    }
    
    /// Calibrate tau_margin to balance precision/recall
    pub fn calibrate_tau_margin(&self) -> f64 {
        if self.known_margins.is_empty() {
            return 0.15; // Default from Section 6
        }
        
        let avg_known: f64 = self.known_margins.iter().sum::<f64>() / self.known_margins.len() as f64;
        let avg_unknown: f64 = if self.unknown_margins.is_empty() {
            0.0
        } else {
            self.unknown_margins.iter().sum::<f64>() / self.unknown_margins.len() as f64
        };
        
        // Set threshold between known and unknown margins
        (avg_known + avg_unknown) / 2.0
    }
}

impl Default for ThresholdCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visual_confidence() {
        let high_conf = vec![0.9, 0.95, 0.88];
        let low_conf = vec![0.3, 0.4, 0.35];
        
        assert!(calculate_visual_confidence(&high_conf) > 0.8);
        assert!(calculate_visual_confidence(&low_conf) < 0.5);
    }

    #[test]
    fn test_gesture_confidence_good_match() {
        let good_match = MatchResult::new(Some("G1".to_string()), 0.2, 0.6, 5);
        let conf = calculate_gesture_confidence(&good_match, 0.5, 0.8);
        
        assert!(conf > 0.7);
    }

    #[test]
    fn test_gesture_confidence_poor_match() {
        let poor_match = MatchResult::new(None, 0.9, 1.0, 0);
        let conf = calculate_gesture_confidence(&poor_match, 0.5, 0.8);
        
        assert!(conf < 0.3);
    }

    #[test]
    fn test_temporal_confidence() {
        // Optimal duration
        assert!(calculate_temporal_confidence(1.0) > 0.9);
        
        // Edge cases (still valid)
        assert!(calculate_temporal_confidence(0.3) > 0.5);
        assert!(calculate_temporal_confidence(2.5) > 0.5);
        
        // Invalid durations
        assert_eq!(calculate_temporal_confidence(0.1), 0.0);
        assert_eq!(calculate_temporal_confidence(3.0), 0.0);
    }

    #[test]
    fn test_context_confidence() {
        // No previous gesture
        assert_eq!(calculate_context_confidence(None, 1000, 500), 1.0);
        
        // Long interval
        assert_eq!(calculate_context_confidence(Some(0), 1000, 500), 1.0);
        
        // Short interval (penalized)
        let conf = calculate_context_confidence(Some(800), 1000, 500);
        assert!(conf < 1.0);
        assert!(conf > 0.0);
    }

    #[test]
    fn test_final_confidence_multiplicative() {
        let factors = ConfidenceFactors {
            c_visual: 0.9,
            c_gesture: 0.8,
            c_temporal: 0.95,
            c_context: 1.0,
            c_symbolic: 0.9,
        };
        
        let c_final = compute_final_confidence(&factors);
        let expected = 0.9 * 0.8 * 0.95 * 1.0 * 0.9;
        
        assert!((c_final - expected).abs() < 1e-6);
    }

    #[test]
    fn test_threshold_calibrator() {
        let mut calibrator = ThresholdCalibrator::new();
        
        // Add known samples (low distances, high margins)
        for _ in 0..10 {
            calibrator.add_known_sample(0.3, 0.4);
        }
        
        // Add unknown samples (high distances, low margins)
        for _ in 0..10 {
            calibrator.add_unknown_sample(0.8, 0.05);
        }
        
        let tau_accept = calibrator.calibrate_tau_accept(0.05);
        assert!(tau_accept > 0.5 && tau_accept < 0.9);
        
        let tau_margin = calibrator.calibrate_tau_margin();
        assert!(tau_margin > 0.1 && tau_margin < 0.3);
    }
}
