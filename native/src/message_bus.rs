//! Message Bus Module for Gestalt-MVP
//! 
//! Defines typed messages for communication between system components.
//! Implements the publish-subscribe pattern for decoupled communication.

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

/// Unique message identifier
pub type MessageId = u64;

/// Gesture identifier (G1-G10)
pub type GestureId = String;

/// Timestamp for audit trail
pub type Timestamp = DateTime<Utc>;

/// Perceptual candidate from MediaPipe hand tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptualCandidate {
    pub timestamp: Timestamp,
    pub hand_present: bool,
    pub landmark_count: u32,
    pub wrist_velocity_energy: f64,
    pub motion_state: MotionState,
}

/// Motion state based on velocity energy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionState {
    Rest,
    Motion,
    Transition,
}

/// Result from DTW matcher
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchResult {
    pub timestamp: Timestamp,
    pub gesture_id: Option<GestureId>,
    pub distance_to_best: f64,      // D1
    pub distance_to_second: f64,    // D2
    pub margin: f64,                 // D2 - D1
    pub exemplar_count: usize,
}

/// Decision from confidence gate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateDecision {
    Accept,
    Clarify,
    Reject,
}

/// Gate decision with full context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateDecisionMessage {
    pub timestamp: Timestamp,
    pub match_result: MatchResult,
    pub decision: GateDecision,
    pub confidence_score: f64,
    pub phrase_template: Option<String>,
}

/// User/caregiver correction input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionMessage {
    pub timestamp: Timestamp,
    pub original_gesture: GestureId,
    pub intended_gesture: Option<GestureId>,
    pub reason: String,
}

/// Control messages for FSM state transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlMessage {
    StartListening,
    StopListening,
    EnterSilentMode,
    ExitSilentMode,
    Reset,
    EmergencyStop,
}

/// Audit trail entry (append-only)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: MessageId,
    pub timestamp: Timestamp,
    pub event_type: String,
    pub details: serde_json::Value,
}

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub timestamp: Timestamp,
    pub camera_ok: bool,
    pub mediapipe_ok: bool,
    pub rust_core_ok: bool,
    pub tts_ok: bool,
    pub fps: f64,
    pub latency_ms: f64,
}

impl PerceptualCandidate {
    pub fn new(
        hand_present: bool,
        landmark_count: u32,
        wrist_velocity_energy: f64,
    ) -> Self {
        let motion_state = if wrist_velocity_energy > 0.0025 {
            MotionState::Motion
        } else if wrist_velocity_energy < 0.001 {
            MotionState::Rest
        } else {
            MotionState::Transition
        };

        Self {
            timestamp: Utc::now(),
            hand_present,
            landmark_count,
            wrist_velocity_energy,
            motion_state,
        }
    }
}

impl MatchResult {
    pub fn new(
        gesture_id: Option<GestureId>,
        distance_to_best: f64,
        distance_to_second: f64,
        exemplar_count: usize,
    ) -> Self {
        let margin = distance_to_second - distance_to_best;
        
        Self {
            timestamp: Utc::now(),
            gesture_id,
            distance_to_best,
            distance_to_second,
            margin,
            exemplar_count,
        }
    }
    
    /// Determine gate decision based on thresholds
    pub fn decide(&self, tau_accept: f64, tau_margin: f64) -> GateDecision {
        if self.distance_to_best > tau_accept {
            GateDecision::Reject
        } else if self.margin > tau_margin {
            GateDecision::Accept
        } else {
            GateDecision::Clarify
        }
    }
}

impl GateDecisionMessage {
    pub fn from_match_result(
        match_result: MatchResult,
        confidence_score: f64,
        phrase_template: Option<String>,
    ) -> Self {
        let decision = match_result.decide(0.5, 0.15); // Default thresholds
        
        Self {
            timestamp: Utc::now(),
            match_result,
            decision,
            confidence_score,
            phrase_template,
        }
    }
}

impl AuditEntry {
    pub fn new(event_type: &str, details: serde_json::Value) -> Self {
        static mut COUNTER: u64 = 0;
        
        unsafe {
            COUNTER += 1;
        }
        
        Self {
            id: unsafe { COUNTER },
            timestamp: Utc::now(),
            event_type: event_type.to_string(),
            details,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_state_detection() {
        let rest_candidate = PerceptualCandidate::new(true, 21, 0.0005);
        assert_eq!(rest_candidate.motion_state, MotionState::Rest);

        let motion_candidate = PerceptualCandidate::new(true, 21, 0.003);
        assert_eq!(motion_candidate.motion_state, MotionState::Motion);
    }

    #[test]
    fn test_gate_decision() {
        let good_match = MatchResult::new(Some("G1".to_string()), 0.3, 0.6, 5);
        assert_eq!(good_match.decide(0.5, 0.15), GateDecision::Accept);

        let ambiguous_match = MatchResult::new(Some("G1".to_string()), 0.3, 0.35, 5);
        assert_eq!(ambiguous_match.decide(0.5, 0.15), GateDecision::Clarify);

        let poor_match = MatchResult::new(None, 0.8, 0.9, 0);
        assert_eq!(poor_match.decide(0.5, 0.15), GateDecision::Reject);
    }
}
