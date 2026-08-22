//! Finite State Machine Module for Gestalt-MVP
//! 
//! Implements the deterministic FSM with states:
//! IDLE, LISTENING, GESTURE_DETECTED, HIGH_CONF, LOW_CONF, 
//! CORRECTION, SILENT, ERROR

use crate::message_bus::{GateDecision, ControlMessage, AuditEntry};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// FSM States as defined in Section 11
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FsmState {
    Idle,
    Listening,
    GestureDetected,
    HighConf,
    LowConf,
    Correction,
    Silent,
    Error,
}

impl Default for FsmState {
    fn default() -> Self {
        FsmState::Idle
    }
}

/// Event types that trigger state transitions
#[derive(Debug, Clone)]
pub enum FsmEvent {
    HandDetected,
    MotionStarted,
    GestureSegmented,
    HighConfidence(GateDecision),
    LowConfidence(GateDecision),
    UserCorrection,
    SoftMuteGesture,
    HardMuteButton,
    SensorFailure,
    RecoverySignal,
    ResetCommand,
}

/// State transition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionResult {
    pub from_state: FsmState,
    pub to_state: FsmState,
    pub event: String,
    pub action: Option<String>,
    pub timestamp: DateTime<Utc>,
}

/// The Finite State Machine
pub struct GestureFsm {
    current_state: FsmState,
    previous_state: FsmState,
    state_history: Vec<TransitionResult>,
    silent_mode: bool,
    error_count: u32,
}

impl GestureFsm {
    pub fn new() -> Self {
        Self {
            current_state: FsmState::Idle,
            previous_state: FsmState::Idle,
            state_history: Vec::new(),
            silent_mode: false,
            error_count: 0,
        }
    }

    /// Get current state
    pub fn current_state(&self) -> FsmState {
        self.current_state
    }

    /// Check if in silent mode
    pub fn is_silent(&self) -> bool {
        self.silent_mode
    }

    /// Process an event and transition state
    pub fn process_event(&mut self, event: FsmEvent) -> Option<TransitionResult> {
        let from_state = self.current_state;
        let mut action: Option<String> = None;
        
        // Determine next state based on current state and event
        let to_state = match (self.current_state, event) {
            // From IDLE
            (FsmState::Idle, FsmEvent::HandDetected) => {
                action = Some("Start listening for gestures".to_string());
                FsmState::Listening
            }
            (FsmState::Idle, FsmEvent::SoftMuteGesture) |
            (FsmState::Idle, FsmEvent::HardMuteButton) => {
                self.silent_mode = true;
                action = Some("Enter silent mode".to_string());
                FsmState::Silent
            }
            (FsmState::Idle, FsmEvent::ResetCommand) => {
                FsmState::Idle
            }

            // From LISTENING
            (FsmState::Listening, FsmEvent::MotionStarted) => {
                action = Some("Motion detected, buffering frames".to_string());
                FsmState::Listening
            }
            (FsmState::Listening, FsmEvent::GestureSegmented) => {
                action = Some("Gesture boundary detected".to_string());
                FsmState::GestureDetected
            }
            (FsmState::Listening, FsmEvent::SensorFailure) => {
                self.error_count += 1;
                action = Some("Sensor failure detected".to_string());
                FsmState::Error
            }
            (FsmState::Listening, FsmEvent::SoftMuteGesture) |
            (FsmState::Listening, FsmEvent::HardMuteButton) => {
                self.silent_mode = true;
                action = Some("Enter silent mode".to_string());
                FsmState::Silent
            }

            // From GESTURE_DETECTED
            (FsmState::GestureDetected, FsmEvent::HighConfidence(GateDecision::Accept)) => {
                action = Some("High confidence match, prepare to speak".to_string());
                FsmState::HighConf
            }
            (FsmState::GestureDetected, FsmEvent::HighConfidence(GateDecision::Clarify)) |
            (FsmState::GestureDetected, FsmEvent::LowConfidence(GateDecision::Clarify)) => {
                action = Some("Ambiguous match, request clarification".to_string());
                FsmState::LowConf
            }
            (FsmState::GestureDetected, FsmEvent::HighConfidence(GateDecision::Reject)) |
            (FsmState::GestureDetected, FsmEvent::LowConfidence(GateDecision::Reject)) => {
                action = Some("Low confidence, reject gesture".to_string());
                FsmState::Listening
            }

            // From HIGH_CONF
            (FsmState::HighConf, FsmEvent::UserCorrection) => {
                action = Some("User correction received".to_string());
                FsmState::Correction
            }
            (FsmState::HighConf, _) => {
                // After speaking, return to listening
                action = Some("Utterance complete, return to listening".to_string());
                FsmState::Listening
            }

            // From LOW_CONF
            (FsmState::LowConf, FsmEvent::UserCorrection) => {
                action = Some("User provided correction".to_string());
                FsmState::Correction
            }
            (FsmState::LowConf, FsmEvent::RecoverySignal) => {
                action = Some("Clarification resolved".to_string());
                FsmState::Listening
            }

            // From CORRECTION
            (FsmState::Correction, FsmEvent::RecoverySignal) => {
                action = Some("Correction applied, resume normal operation".to_string());
                FsmState::Listening
            }

            // From SILENT
            (FsmState::Silent, FsmEvent::SoftMuteGesture) |
            (FsmState::Silent, FsmEvent::HardMuteButton) => {
                self.silent_mode = false;
                action = Some("Exit silent mode".to_string());
                FsmState::Idle
            }
            (FsmState::Silent, FsmEvent::EmergencyStop) => {
                action = Some("Emergency stop while silent".to_string());
                FsmState::Error
            }

            // From ERROR
            (FsmState::Error, FsmEvent::RecoverySignal) => {
                self.error_count = 0;
                action = Some("System recovered from error".to_string());
                FsmState::Idle
            }
            (FsmState::Error, FsmEvent::ResetCommand) => {
                self.error_count = 0;
                action = Some("Manual reset from error state".to_string());
                FsmState::Idle
            }

            // Default: stay in current state
            (_, _) => {
                self.current_state
            }
        };

        // Record transition if state changed
        if to_state != from_state {
            self.previous_state = from_state;
            self.current_state = to_state;
            
            let event_name = match &event {
                FsmEvent::HandDetected => "HandDetected",
                FsmEvent::MotionStarted => "MotionStarted",
                FsmEvent::GestureSegmented => "GestureSegmented",
                FsmEvent::HighConfidence(d) => format!("HighConfidence({:?})", d),
                FsmEvent::LowConfidence(d) => format!("LowConfidence({:?})", d),
                FsmEvent::UserCorrection => "UserCorrection",
                FsmEvent::SoftMuteGesture => "SoftMuteGesture",
                FsmEvent::HardMuteButton => "HardMuteButton",
                FsmEvent::SensorFailure => "SensorFailure",
                FsmEvent::RecoverySignal => "RecoverySignal",
                FsmEvent::ResetCommand => "ResetCommand",
                FsmEvent::EmergencyStop => "EmergencyStop",
            };

            let result = TransitionResult {
                from_state,
                to_state,
                event: event_name.to_string(),
                action: action.clone(),
                timestamp: Utc::now(),
            };

            self.state_history.push(result.clone());
            Some(result)
        } else {
            None
        }
    }

    /// Get state history for audit trail
    pub fn get_history(&self) -> &[TransitionResult] {
        &self.state_history
    }

    /// Reset FSM to initial state
    pub fn reset(&mut self) {
        self.current_state = FsmState::Idle;
        self.previous_state = FsmState::Idle;
        self.silent_mode = false;
        self.error_count = 0;
    }

    /// Create audit entry for current state
    pub fn create_audit_entry(&self) -> AuditEntry {
        use serde_json::json;
        
        let details = json!({
            "current_state": format!("{:?}", self.current_state),
            "previous_state": format!("{:?}", self.previous_state),
            "silent_mode": self.silent_mode,
            "error_count": self.error_count,
            "history_length": self.state_history.len(),
        });

        AuditEntry::new("fsm_state_update", details)
    }

    /// Check if system is operational (not in error state)
    pub fn is_operational(&self) -> bool {
        self.current_state != FsmState::Error
    }

    /// Get error count
    pub fn error_count(&self) -> u32 {
        self.error_count
    }
}

impl Default for GestureFsm {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let fsm = GestureFsm::new();
        assert_eq!(fsm.current_state(), FsmState::Idle);
        assert!(!fsm.is_silent());
        assert!(fsm.is_operational());
    }

    #[test]
    fn test_idle_to_listening_transition() {
        let mut fsm = GestureFsm::new();
        
        let result = fsm.process_event(FsmEvent::HandDetected);
        assert!(result.is_some());
        assert_eq!(fsm.current_state(), FsmState::Listening);
        assert_eq!(result.unwrap().action, Some("Start listening for gestures".to_string()));
    }

    #[test]
    fn test_full_gesture_recognition_flow() {
        let mut fsm = GestureFsm::new();
        
        // Idle → Listening
        fsm.process_event(FsmEvent::HandDetected);
        assert_eq!(fsm.current_state(), FsmState::Listening);
        
        // Listening → GestureDetected
        fsm.process_event(FsmEvent::GestureSegmented);
        assert_eq!(fsm.current_state(), FsmState::GestureDetected);
        
        // GestureDetected → HighConf (accept)
        fsm.process_event(FsmEvent::HighConfidence(GateDecision::Accept));
        assert_eq!(fsm.current_state(), FsmState::HighConf);
        
        // HighConf → Listening (after utterance)
        fsm.process_event(FsmEvent::RecoverySignal);
        assert_eq!(fsm.current_state(), FsmState::Listening);
    }

    #[test]
    fn test_silent_mode_transitions() {
        let mut fsm = GestureFsm::new();
        
        // Enter silent mode
        fsm.process_event(FsmEvent::HardMuteButton);
        assert_eq!(fsm.current_state(), FsmState::Silent);
        assert!(fsm.is_silent());
        
        // Exit silent mode
        fsm.process_event(FsmEvent::HardMuteButton);
        assert_eq!(fsm.current_state(), FsmState::Idle);
        assert!(!fsm.is_silent());
    }

    #[test]
    fn test_error_handling() {
        let mut fsm = GestureFsm::new();
        fsm.process_event(FsmEvent::HandDetected);
        
        // Simulate sensor failure
        fsm.process_event(FsmEvent::SensorFailure);
        assert_eq!(fsm.current_state(), FsmState::Error);
        assert_eq!(fsm.error_count(), 1);
        assert!(!fsm.is_operational());
        
        // Recover
        fsm.process_event(FsmEvent::RecoverySignal);
        assert_eq!(fsm.current_state(), FsmState::Idle);
        assert!(fsm.is_operational());
        assert_eq!(fsm.error_count(), 0);
    }

    #[test]
    fn test_correction_flow() {
        let mut fsm = GestureFsm::new();
        
        fsm.process_event(FsmEvent::HandDetected);
        fsm.process_event(FsmEvent::GestureSegmented);
        fsm.process_event(FsmEvent::HighConfidence(GateDecision::Accept));
        assert_eq!(fsm.current_state(), FsmState::HighConf);
        
        // User indicates wrong recognition
        fsm.process_event(FsmEvent::UserCorrection);
        assert_eq!(fsm.current_state(), FsmState::Correction);
        
        // Apply correction and recover
        fsm.process_event(FsmEvent::RecoverySignal);
        assert_eq!(fsm.current_state(), FsmState::Listening);
    }
}
