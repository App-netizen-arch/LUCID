//! Correction Loop Module for Gestalt-MVP
//! 
//! Implements the correction mechanism that modifies the MAPPING
//! (gesture_id → meaning), never overwrites the perceptual prototype.
//! 
//! Key principles:
//! - Keep ALL exemplars. Never replace class with single sample.
//! - Corrections are logged to the audit trail.
//! - Uses DTW Barycenter Averaging (DBA) if compact representation needed.

use crate::graph::{GesturePrototype, Mapping};
use crate::message_bus::{CorrectionMessage, AuditEvent};
use std::collections::HashMap;

/// Correction state machine
#[derive(Debug, Clone, PartialEq)]
pub enum CorrectionState {
    Idle,
    PendingCorrection(GestureId, String), // gesture_id, new_meaning
    Applied,
}

#[derive(Debug, Clone)]
pub struct GestureId(pub String);

/// Correction handler
pub struct CorrectionHandler {
    state: CorrectionState,
    correction_count: u32,
}

impl CorrectionHandler {
    pub fn new() -> Self {
        Self {
            state: CorrectionState::Idle,
            correction_count: 0,
        }
    }
    
    /// Apply a correction to the mapping
    /// 
    /// This modifies which meaning is associated with a gesture,
    /// but does NOT change the gesture prototypes/exemplars.
    pub fn apply_correction(
        &mut self,
        gesture_id: &str,
        new_meaning: String,
        mappings: &mut HashMap<String, Mapping>,
    ) -> Result<CorrectionMessage, String> {
        // Verify gesture exists
        if !mappings.contains_key(gesture_id) {
            return Err(format!("Gesture {} not found in mappings", gesture_id));
        }
        
        // Store old meaning for audit
        let old_meaning = mappings[gesture_id].meaning.clone();
        
        // Update the mapping (not the prototype!)
        if let Some(mapping) = mappings.get_mut(gesture_id) {
            mapping.meaning = new_meaning.clone();
            mapping.is_customized = true;
        }
        
        self.correction_count += 1;
        self.state = CorrectionState::Applied;
        
        let message = CorrectionMessage {
            gesture_id: gesture_id.to_string(),
            old_meaning,
            new_meaning,
            correction_index: self.correction_count,
            timestamp: chrono::Utc::now(),
        };
        
        Ok(message)
    }
    
    /// Get current correction state
    pub fn get_state(&self) -> &CorrectionState {
        &self.state
    }
    
    /// Reset correction state
    pub fn reset(&mut self) {
        self.state = CorrectionState::Idle;
    }
    
    /// Get total correction count
    pub fn correction_count(&self) -> u32 {
        self.correction_count
    }
}

impl Default for CorrectionHandler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_apply_correction() {
        let mut handler = CorrectionHandler::new();
        let mut mappings = HashMap::new();
        
        mappings.insert(
            "G1".to_string(),
            Mapping {
                gesture_id: "G1".to_string(),
                meaning: "water".to_string(),
                phrase_template: "I would like water.".to_string(),
                is_customized: false,
            },
        );
        
        let result = handler.apply_correction(
            "G1",
            "cold water".to_string(),
            &mut mappings,
        );
        
        assert!(result.is_ok());
        assert_eq!(mappings["G1"].meaning, "cold water");
        assert!(mappings["G1"].is_customized);
    }
}
