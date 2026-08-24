//! Guards Module for Gestalt-MVP
//! 
//! Implements three safety guards:
//! - Physics Guard: Validates motion is physically plausible
//! - Social Guard: Filters inappropriate gestures in context
//! - Epistemic Guard: Ensures system knows its limitations

use crate::graph::Trajectory;
use crate::message_bus::{MatchResult, GateDecision};

/// Physics validation parameters
pub struct PhysicsParams {
    /// Maximum plausible hand velocity (m/s)
    pub max_velocity: f64,
    /// Maximum plausible hand acceleration (m/s²)
    pub max_acceleration: f64,
    /// Minimum distance between landmarks (m)
    pub min_landmark_distance: f64,
    /// Maximum distance between landmarks (m)
    pub max_landmark_distance: f64,
}

impl Default for PhysicsParams {
    fn default() -> Self {
        Self {
            // Assuming normalized coordinates where 1.0 ≈ arm's length
            max_velocity: 5.0,        // ~5 m/s is very fast hand movement
            max_acceleration: 50.0,   // ~50 m/s² is extreme acceleration
            min_landmark_distance: 0.001, // 1mm minimum
            max_landmark_distance: 0.5,   // 50cm maximum span
        }
    }
}

/// Social guard configuration
pub struct SocialParams {
    /// Gestures allowed in current context
    pub allowed_gestures: Vec<String>,
    /// Time-based restrictions (e.g., no "bathroom" at meal times)
    pub time_restricted_gestures: Vec<(String, u8, u8)>, // (gesture, start_hour, end_hour)
}

impl Default for SocialParams {
    fn default() -> Self {
        Self {
            allowed_gestures: vec![
                "G1".to_string(), "G2".to_string(), "G3".to_string(),
                "G4".to_string(), "G5".to_string(), "G6".to_string(),
                "G7".to_string(), "G8".to_string(), "G9".to_string(),
                "G10".to_string(),
            ],
            time_restricted_gestures: Vec::new(),
        }
    }
}

/// Epistemic guard configuration
pub struct EpistemicParams {
    /// Minimum exemplars required per class
    pub min_exemplars_per_class: usize,
    /// Maximum acceptable class imbalance ratio
    pub max_imbalance_ratio: f64,
    /// Confidence threshold for "unknown" classification
    pub unknown_threshold: f64,
}

impl Default for EpistemicParams {
    fn default() -> Self {
        Self {
            min_exemplars_per_class: 5,  // From Section 6
            max_imbalance_ratio: 3.0,     // No class should have 3× more exemplars
            unknown_threshold: 0.8,       // High bar for claiming "unknown"
        }
    }
}

/// Result of a guard check
#[derive(Debug, Clone)]
pub struct GuardResult {
    pub passed: bool,
    pub guard_type: GuardType,
    pub reason: Option<String>,
    pub severity: Severity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardType {
    Physics,
    Social,
    Epistemic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Info,
    Warning,
    Critical,
}

/// Physics Guard: Validates physical plausibility of trajectory
pub struct PhysicsGuard {
    params: PhysicsParams,
}

impl PhysicsGuard {
    pub fn new(params: PhysicsParams) -> Self {
        Self { params }
    }

    pub fn validate_trajectory(&self, trajectory: &Trajectory) -> GuardResult {
        if trajectory.frames.len() < 2 {
            return GuardResult {
                passed: false,
                guard_type: GuardType::Physics,
                reason: Some("Insufficient frames for physics validation".to_string()),
                severity: Severity::Critical,
            };
        }

        // Check velocities
        for i in 1..trajectory.frames.len() {
            let prev = &trajectory.frames[i - 1];
            let curr = &trajectory.frames[i];
            let dt = curr.t - prev.t;

            if dt <= 0.0 {
                continue;
            }

            // Calculate wrist velocity
            let prev_wrist = prev.landmarks[0];
            let curr_wrist = curr.landmarks[0];

            let dx = (curr_wrist[0] - prev_wrist[0]) as f64 / dt;
            let dy = (curr_wrist[1] - prev_wrist[1]) as f64 / dt;
            let dz = (curr_wrist[2] - prev_wrist[2]) as f64 / dt;

            let velocity = (dx * dx + dy * dy + dz * dz).sqrt();

            if velocity > self.params.max_velocity {
                return GuardResult {
                    passed: false,
                    guard_type: GuardType::Physics,
                    reason: Some(format!(
                        "Velocity {} m/s exceeds maximum {}",
                        velocity, self.params.max_velocity
                    )),
                    severity: Severity::Warning,
                };
            }
        }

        // Check landmark distances (hand structure validity)
        for frame in &trajectory.frames {
            for i in 0..frame.landmarks.len() {
                for j in (i + 1)..frame.landmarks.len() {
                    let lm_i = frame.landmarks[i];
                    let lm_j = frame.landmarks[j];

                    let dx = (lm_j[0] - lm_i[0]) as f64;
                    let dy = (lm_j[1] - lm_i[1]) as f64;
                    let dz = (lm_j[2] - lm_i[2]) as f64;

                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                    if dist < self.params.min_landmark_distance {
                        return GuardResult {
                            passed: false,
                            guard_type: GuardType::Physics,
                            reason: Some("Landmarks too close together".to_string()),
                            severity: Severity::Warning,
                        };
                    }

                    if dist > self.params.max_landmark_distance {
                        return GuardResult {
                            passed: false,
                            guard_type: GuardType::Physics,
                            reason: Some("Landmarks too far apart".to_string()),
                            severity: Severity::Warning,
                        };
                    }
                }
            }
        }

        GuardResult {
            passed: true,
            guard_type: GuardType::Physics,
            reason: None,
            severity: Severity::Info,
        }
    }
}

impl Default for PhysicsGuard {
    fn default() -> Self {
        Self::new(PhysicsParams::default())
    }
}

/// Social Guard: Contextual appropriateness filter
pub struct SocialGuard {
    params: SocialParams,
}

impl SocialGuard {
    pub fn new(params: SocialParams) -> Self {
        Self { params }
    }

    pub fn validate_gesture(&self, gesture_id: &str, current_hour: Option<u8>) -> GuardResult {
        // Check if gesture is allowed
        if !self.params.allowed_gestures.iter().any(|g| g == gesture_id) {
            return GuardResult {
                passed: false,
                guard_type: GuardType::Social,
                reason: Some(format!("Gesture {} not allowed in current context", gesture_id)),
                severity: Severity::Warning,
            };
        }

        // Check time-based restrictions
        if let Some(hour) = current_hour {
            for (restricted_gesture, start, end) in &self.params.time_restricted_gestures {
                if restricted_gesture == gesture_id {
                    if hour >= *start && hour < *end {
                        return GuardResult {
                            passed: false,
                            guard_type: GuardType::Social,
                            reason: Some(format!(
                                "Gesture {} restricted during hours {}-{}",
                                gesture_id, start, end
                            )),
                            severity: Severity::Info,
                        };
                    }
                }
            }
        }

        GuardResult {
            passed: true,
            guard_type: GuardType::Social,
            reason: None,
            severity: Severity::Info,
        }
    }

    /// Update allowed gestures dynamically
    pub fn set_allowed_gestures(&mut self, gestures: Vec<String>) {
        self.params.allowed_gestures = gestures;
    }
}

impl Default for SocialGuard {
    fn default() -> Self {
        Self::new(SocialParams::default())
    }
}

/// Epistemic Guard: System knowledge validation
pub struct EpistemicGuard {
    params: EpistemicParams,
}

impl EpistemicGuard {
    pub fn new(params: EpistemicParams) -> Self {
        Self { params }
    }

    pub fn validate_enrollment(
        &self,
        exemplar_counts: &[usize],
    ) -> GuardResult {
        if exemplar_counts.is_empty() {
            return GuardResult {
                passed: false,
                guard_type: GuardType::Epistemic,
                reason: Some("No gesture classes enrolled".to_string()),
                severity: Severity::Critical,
            };
        }

        // Check minimum exemplars per class
        for (i, &count) in exemplar_counts.iter().enumerate() {
            if count < self.params.min_exemplars_per_class {
                return GuardResult {
                    passed: false,
                    guard_type: GuardType::Epistemic,
                    reason: Some(format!(
                        "Class {} has {} exemplars, minimum is {}",
                        i, count, self.params.min_exemplars_per_class
                    )),
                    severity: Severity::Warning,
                };
            }
        }

        // Check class imbalance
        let max_count = *exemplar_counts.iter().max().unwrap_or(&0);
        let min_count = *exemplar_counts.iter().min().unwrap_or(&1);

        if min_count > 0 {
            let ratio = max_count as f64 / min_count as f64;
            if ratio > self.params.max_imbalance_ratio {
                return GuardResult {
                    passed: false,
                    guard_type: GuardType::Epistemic,
                    reason: Some(format!(
                        "Class imbalance ratio {} exceeds maximum {}",
                        ratio, self.params.max_imbalance_ratio
                    )),
                    severity: Severity::Warning,
                };
            }
        }

        GuardResult {
            passed: true,
            guard_type: GuardType::Epistemic,
            reason: None,
            severity: Severity::Info,
        }
    }

    pub fn validate_match_confidence(&self, match_result: &MatchResult) -> GuardResult {
        // Check if we're confident enough to claim "unknown"
        if match_result.distance_to_best > self.params.unknown_threshold {
            GuardResult {
                passed: true, // It's okay to reject as unknown
                guard_type: GuardType::Epistemic,
                reason: Some("Input too distant from all known classes".to_string()),
                severity: Severity::Info,
            }
        } else if match_result.margin <= 0.0 {
            GuardResult {
                passed: false,
                guard_type: GuardType::Epistemic,
                reason: Some("Cannot distinguish between classes (negative margin)".to_string()),
                severity: Severity::Warning,
            }
        } else {
            GuardResult {
                passed: true,
                guard_type: GuardType::Epistemic,
                reason: None,
                severity: Severity::Info,
            }
        }
    }
}

impl Default for EpistemicGuard {
    fn default() -> Self {
        Self::new(EpistemicParams::default())
    }
}

/// Combined guard evaluation
pub fn run_all_guards(
    trajectory: &Trajectory,
    gesture_id: Option<&str>,
    exemplar_counts: &[usize],
    match_result: Option<&MatchResult>,
    current_hour: Option<u8>,
) -> Vec<GuardResult> {
    let mut results = Vec::new();

    // Physics guard
    let physics_guard = PhysicsGuard::default();
    results.push(physics_guard.validate_trajectory(trajectory));

    // Social guard (if gesture identified)
    if let Some(gid) = gesture_id {
        let social_guard = SocialGuard::default();
        results.push(social_guard.validate_gesture(gid, current_hour));
    }

    // Epistemic guard
    let epistemic_guard = EpistemicGuard::default();
    results.push(epistemic_guard.validate_enrollment(exemplar_counts));

    if let Some(mr) = match_result {
        results.push(epistemic_guard.validate_match_confidence(mr));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Frame;

    fn create_test_trajectory() -> Trajectory {
        let frames = vec![
            Frame {
                t: 0.0,
                landmarks: vec![[0.0, 0.0, 0.0]; 21],
            },
            Frame {
                t: 0.033,
                landmarks: vec![[0.01, 0.01, 0.01]; 21],
            },
        ];
        Trajectory {
            frames,
            duration_s: 0.033,
            frame_count: 2,
        }
    }

    #[test]
    fn test_physics_guard_valid_trajectory() {
        let guard = PhysicsGuard::default();
        let trajectory = create_test_trajectory();
        let result = guard.validate_trajectory(&trajectory);

        assert!(result.passed);
        assert_eq!(result.guard_type, GuardType::Physics);
    }

    #[test]
    fn test_social_guard_allowed_gesture() {
        let guard = SocialGuard::default();
        let result = guard.validate_gesture("G1", None);

        assert!(result.passed);
    }

    #[test]
    fn test_social_guard_disallowed_gesture() {
        let mut guard = SocialGuard::default();
        guard.params.allowed_gestures = vec!["G1".to_string(), "G2".to_string()];

        let result = guard.validate_gesture("G5", None);

        assert!(!result.passed);
        assert_eq!(result.guard_type, GuardType::Social);
    }

    #[test]
    fn test_epistemic_guard_sufficient_exemplars() {
        let guard = EpistemicGuard::default();
        let counts = vec![5, 5, 5, 5, 5, 5, 5, 5, 5, 5];
        let result = guard.validate_enrollment(&counts);

        assert!(result.passed);
    }

    #[test]
    fn test_epistemic_guard_insufficient_exemplars() {
        let guard = EpistemicGuard::default();
        let counts = vec![5, 5, 3, 5, 5, 5, 5, 5, 5, 5];
        let result = guard.validate_enrollment(&counts);

        assert!(!result.passed);
        assert_eq!(result.guard_type, GuardType::Epistemic);
    }

    #[test]
    fn test_combined_guards() {
        let trajectory = create_test_trajectory();
        let exemplar_counts = vec![5; 10];

        let results = run_all_guards(
            &trajectory,
            Some("G1"),
            &exemplar_counts,
            None,
            None,
        );

        assert!(!results.is_empty());
        // Physics should pass, epistemic enrollment should pass
    }
}
