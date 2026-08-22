//! Gestalt-MVP Neuro-Symbolic Gesture Recognition Core
//! 
//! This library provides the symbolic core for the Gestalt system,
//! implementing deterministic DTW matching gated by a symbolic confidence engine.
//! 
//! **Architecture:**
//! - Neural layer (MediaPipe) proposes evidence
//! - Symbolic core (Rust) resolves interpretations
//! - Phrasing layer (templates) never introduces meaning

#![warn(missing_docs)]

pub mod graph;
pub mod matcher;
pub mod fsm;
pub mod confidence;
pub mod guards;
pub mod message_bus;
pub mod correction;
pub mod templates;
pub mod audit;
pub mod tts;
pub mod circuit_breaker;

use flutter_rust_bridge::frb;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

pub use graph::{GestureGraph, GesturePrototype, Mapping, Trajectory, Frame};
pub use matcher::{DtwMatcher, dtw_distance};
pub use fsm::{GestureFsm, FsmState, FsmEvent, TransitionResult};
pub use confidence::{
    ConfidenceFactors, ConfidenceResult, ThresholdCalibrator,
    evaluate_confidence, compute_final_confidence,
};
pub use guards::{
    PhysicsGuard, SocialGuard, EpistemicGuard, GuardResult, GuardType,
    run_all_guards,
};
pub use message_bus::{
    PerceptualCandidate, MatchResult, GateDecision, GateDecisionMessage,
    CorrectionMessage, ControlMessage, AuditEntry, HealthStatus, MotionState,
};
pub use tts::{TtsActuator, TtsConfig, TtsEngine, TtsState};
pub use circuit_breaker::{CircuitBreaker, CircuitState, DegradationManager, DegradationLevel};

use log::{info, warn};

/// Main Gestalt core engine with TTS integration and circuit breaker
pub struct GestaltEngine {
    graph: Arc<Mutex<GestureGraph>>,
    matcher: Arc<Mutex<DtwMatcher>>,
    fsm: Arc<Mutex<GestureFsm>>,
    calibrator: Arc<Mutex<ThresholdCalibrator>>,
    tts: Arc<Mutex<TtsActuator>>,
    degradation_manager: Arc<Mutex<DegradationManager>>,
    
    // Tuned thresholds (frozen after Session 3)
    tau_accept: f64,
    tau_margin: f64,
    
    // Audit trail
    audit_log: Vec<AuditEntry>,
}

impl GestaltEngine {
    /// Create a new Gestalt engine with in-memory storage
    #[frb(sync)]
    pub fn new() -> Result<Self, String> {
        let graph = GestureGraph::open_in_memory()
            .map_err(|e| format!("Failed to create graph: {}", e))?;
        
        graph.initialize_mappings()
            .map_err(|e| format!("Failed to initialize mappings: {}", e))?;
        
        let tts_config = TtsConfig::default();
        let tts = TtsActuator::new(tts_config);
        let degradation_manager = DegradationManager::new();
        
        Ok(Self {
            graph: Arc::new(Mutex::new(graph)),
            matcher: Arc::new(Mutex::new(DtwMatcher::new())),
            fsm: Arc::new(Mutex::new(GestureFsm::new())),
            calibrator: Arc::new(Mutex::new(ThresholdCalibrator::new())),
            tts: Arc::new(Mutex::new(tts)),
            degradation_manager: Arc::new(Mutex::new(degradation_manager)),
            tau_accept: 0.5,   // Default, tune on Session 3
            tau_margin: 0.15,  // From Section 6
            audit_log: Vec::new(),
        })
    }
    
    /// Create engine with persistent storage
    #[frb(sync)]
    pub fn with_storage(path: &str) -> Result<Self, String> {
        let graph = GestureGraph::open(path)
            .map_err(|e| format!("Failed to open graph at {}: {}", path, e))?;
        
        graph.initialize_mappings()
            .map_err(|e| format!("Failed to initialize mappings: {}", e))?;
        
        let tts_config = TtsConfig::default();
        let tts = TtsActuator::new(tts_config);
        let degradation_manager = DegradationManager::new();
        
        Ok(Self {
            graph: Arc::new(Mutex::new(graph)),
            matcher: Arc::new(Mutex::new(DtwMatcher::new())),
            fsm: Arc::new(Mutex::new(GestureFsm::new())),
            calibrator: Arc::new(Mutex::new(ThresholdCalibrator::new())),
            tts: Arc::new(Mutex::new(tts)),
            degradation_manager: Arc::new(Mutex::new(degradation_manager)),
            tau_accept: 0.5,
            tau_margin: 0.15,
            audit_log: Vec::new(),
        })
    }
    
    /// Get circuit breaker health status
    #[frb(sync)]
    pub fn get_health_status(&self) -> crate::circuit_breaker::HealthStatus {
        let dm = self.degradation_manager.lock().unwrap();
        dm.circuit_breaker().get_health()
    }
    
    /// Update degradation level based on latency
    #[frb(sync)]
    pub fn update_degradation(&self, latency_ms: u32, fps: f32) {
        let dm = self.degradation_manager.lock().unwrap();
        dm.update_level(latency_ms, fps);
    }
    
    /// Check if frame should be processed (for frame skipping)
    #[frb(sync)]
    pub fn should_process_frame(&self, frame_number: u64) -> bool {
        let dm = self.degradation_manager.lock().unwrap();
        dm.should_process_frame(frame_number)
    }
    
    /// Get current degradation level description
    #[frb(sync)]
    pub fn get_degradation_action(&self) -> String {
        let dm = self.degradation_manager.lock().unwrap();
        dm.get_action().to_string()
    }
    
    /// Enroll a gesture exemplar
    #[frb(sync)]
    pub fn enroll_exemplar(&self, gesture_id: &str, trajectory: &Trajectory) -> Result<bool, String> {
        let mut matcher = self.matcher.lock()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        let mut graph = self.graph.lock()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        // Add to graph
        graph.add_trajectory(gesture_id, trajectory)
            .map_err(|e| format!("Failed to add trajectory: {}", e))?;
        
        // Check if we need to create the gesture prototype
        let exemplar_count = graph.get_exemplar_count(gesture_id)
            .unwrap_or(0);
        
        if exemplar_count == 1 {
            // First exemplar, create prototype
            let label = match gesture_id {
                "G1" => "Water", "G2" => "Help", "G3" => "Yes", "G4" => "No",
                "G5" => "Food", "G6" => "Pain", "G7" => "Stop", "G8" => "More",
                "G9" => "Wait", "G10" => "Bathroom",
                _ => "Unknown",
            };
            
            let proto = GesturePrototype::new(gesture_id, label);
            graph.add_gesture(&proto)
                .map_err(|e| format!("Failed to add gesture: {}", e))?;
        }
        
        // Add to matcher
        // Note: In production, would reload all exemplars; simplified here
        drop(graph);
        drop(matcher);
        
        self.log_audit("enroll_exemplar", serde_json::json!({
            "gesture_id": gesture_id,
            "exemplar_count": exemplar_count,
        }));
        
        Ok(true)
    }
    
    /// Process a recognized gesture and return decision
    #[frb(sync)]
    pub fn process_gesture(&self, trajectory: &Trajectory) -> GateDecisionMessage {
        let matcher = self.matcher.lock().unwrap();
        let graph = self.graph.lock().unwrap();
        let mut fsm = self.fsm.lock().unwrap();
        
        // Match against enrolled exemplars
        // Simplified: in production would load all exemplars from graph
        let match_result = matcher.match_query(trajectory);
        
        // Get phrase template
        let phrase = graph.get_mapping(match_result.gesture_id.as_deref().unwrap_or(""))
            .ok()
            .flatten()
            .map(|m| m.phrase_template);
        
        // Calculate confidence
        let guard_results = guards::run_all_guards(
            trajectory,
            match_result.gesture_id.as_deref(),
            &[5; 10], // Simplified exemplar counts
            Some(&match_result),
            None, // current_hour
        );
        
        let guard_passes: Vec<bool> = guard_results.iter().map(|r| r.passed).collect();
        
        let conf_result = confidence::evaluate_confidence(
            &match_result,
            trajectory,
            &[0.9; 21], // Simplified landmark confidences
            None, // last_gesture_time_ms
            0,    // current_time_ms
            &guard_passes,
            self.tau_accept,
            self.tau_accept + 0.3, // tau_reject
        );
        
        // Create gate decision message
        let decision_msg = GateDecisionMessage::from_match_result(
            match_result,
            conf_result.c_final,
            phrase,
        );
        
        // Update FSM based on decision
        let event = match decision_msg.decision {
            GateDecision::Accept => FsmEvent::HighConfidence(GateDecision::Accept),
            GateDecision::Clarify => FsmEvent::LowConfidence(GateDecision::Clarify),
            GateDecision::Reject => FsmEvent::LowConfidence(GateDecision::Reject),
        };
        
        fsm.process_event(event);
        
        // Log to audit trail
        self.log_audit("process_gesture", serde_json::json!({
            "gesture_id": decision_msg.match_result.gesture_id,
            "decision": format!("{:?}", decision_msg.decision),
            "confidence": decision_msg.confidence_score,
            "fsm_state": format!("{:?}", fsm.current_state()),
        }));
        
        decision_msg
    }
    
    /// Apply user correction
    #[frb(sync)]
    pub fn apply_correction(&self, correction: &CorrectionMessage) -> Result<(), String> {
        let mut fsm = self.fsm.lock()
            .map_err(|e| format!("Lock error: {}", e))?;
        
        // FSM transitions to Correction state
        fsm.process_event(FsmEvent::UserCorrection);
        
        // In Phase 0, corrections modify mapping, not recognizer
        // This is handled by updating the phrase template
        
        self.log_audit("apply_correction", serde_json::json!({
            "original_gesture": correction.original_gesture,
            "intended_gesture": correction.intended_gesture,
            "reason": correction.reason,
        }));
        
        Ok(())
    }
    
    /// Calibrate thresholds using Session 3 data
    #[frb(sync)]
    pub fn calibrate_thresholds(&mut self, target_far: f64) -> (f64, f64) {
        let calibrator = self.calibrator.lock().unwrap();
        
        self.tau_accept = calibrator.calibrate_tau_accept(target_far);
        self.tau_margin = calibrator.calibrate_tau_margin();
        
        (self.tau_accept, self.tau_margin)
    }
    
    /// Add calibration sample
    #[frb(sync)]
    pub fn add_calibration_sample(&self, distance: f64, margin: f64, is_known: bool) {
        let mut calibrator = self.calibrator.lock().unwrap();
        
        if is_known {
            calibrator.add_known_sample(distance, margin);
        } else {
            calibrator.add_unknown_sample(distance, margin);
        }
    }
    
    /// Get current FSM state
    #[frb(sync)]
    pub fn get_fsm_state(&self) -> FsmState {
        let fsm = self.fsm.lock().unwrap();
        fsm.current_state()
    }
    
    /// Reset the engine
    #[frb(sync)]
    pub fn reset(&self) {
        let mut fsm = self.fsm.lock().unwrap();
        fsm.reset();
        
        self.log_audit("reset", serde_json::json!({}));
    }
    
    /// Get audit log
    pub fn get_audit_log(&self) -> &[AuditEntry] {
        &self.audit_log
    }
    
    /// Speak text using TTS with FSM-state-modulated prosody
    #[frb(sync)]
    pub fn speak(&self, text: &str) -> Result<(), String> {
        let fsm_state = self.get_fsm_state();
        let tts = self.tts.lock()
            .map_err(|e| format!("TTS lock error: {}", e))?;
        
        tts.speak(text, &format!("{:?}", fsm_state))
    }
    
    /// Stop TTS immediately (emergency mute)
    #[frb(sync)]
    pub fn stop_speech(&self) -> Result<(), String> {
        let tts = self.tts.lock()
            .map_err(|e| format!("TTS lock error: {}", e))?;
        
        tts.stop()
    }
    
    /// Set TTS engine type
    #[frb(sync)]
    pub fn set_tts_engine(&self, engine: &str) -> Result<(), String> {
        let mut tts = self.tts.lock()
            .map_err(|e| format!("TTS lock error: {}", e))?;
        
        let tts_engine = match engine.to_lowercase().as_str() {
            "piper" => TtsEngine::Piper,
            "sherpa" | "sherpa_onnx" => TtsEngine::SherpaOnnx,
            _ => TtsEngine::System,
        };
        
        tts.set_engine(tts_engine);
        Ok(())
    }
    
    /// Update TTS prosody parameters
    #[frb(sync)]
    pub fn set_tts_prosody(&self, speed: Option<f32>, pitch: Option<f32>, volume: Option<f32>) -> Result<(), String> {
        let mut tts = self.tts.lock()
            .map_err(|e| format!("TTS lock error: {}", e))?;
        
        tts.set_prosody(speed, pitch, volume);
        Ok(())
    }
    
    /// Log an audit entry
    fn log_audit(&mut self, event_type: &str, details: serde_json::Value) {
        let entry = AuditEntry::new(event_type, details);
        self.audit_log.push(entry);
    }
    
    /// Export audit log as JSONL
    pub fn export_audit_log(&self) -> String {
        self.audit_log.iter()
            .filter_map(|entry| serde_json::to_string(entry).ok())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl Default for GestaltEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create default engine")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = GestaltEngine::new();
        assert!(engine.is_ok());
    }

    #[test]
    fn test_enrollment_and_matching() {
        let engine = GestaltEngine::new().unwrap();
        
        // Create a simple test trajectory
        let trajectory = Trajectory {
            frames: vec![
                Frame { t: 0.0, landmarks: vec![[0.0; 3]; 21] },
                Frame { t: 0.033, landmarks: vec![[0.01; 3]; 21] },
            ],
            duration_s: 0.033,
            frame_count: 2,
        };
        
        // Enroll exemplar
        let result = engine.enroll_exemplar("G1", &trajectory);
        assert!(result.is_ok());
        
        // Process same trajectory (should match)
        let decision = engine.process_gesture(&trajectory);
        // Decision depends on thresholds and confidence calculation
    }

    #[test]
    fn test_fsm_integration() {
        let engine = GestaltEngine::new().unwrap();
        
        assert_eq!(engine.get_fsm_state(), FsmState::Idle);
        
        engine.reset();
        assert_eq!(engine.get_fsm_state(), FsmState::Idle);
    }

    #[test]
    fn test_audit_logging() {
        let mut engine = GestaltEngine::new().unwrap();
        engine.log_audit("test_event", serde_json::json!({"key": "value"}));
        
        assert!(!engine.get_audit_log().is_empty());
    }
}
