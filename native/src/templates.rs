//! Phrase Template Engine for Gestalt-MVP
//! 
//! Deterministic string interpolation only. NO LLM, NO Qwen.
//! Each gesture maps to a fixed template with optional slot filling.

use std::collections::HashMap;

/// Fixed phrase templates for the 10-gesture vocabulary
/// Per specification Section 8 - FROZEN
pub const TEMPLATES: &[(&str, &str)] = &[
    ("G1", "I would like some water, please."),
    ("G2", "Could you help me, please?"),
    ("G3", "Yes."),
    ("G4", "No."),
    ("G5", "I would like some food, please."),
    ("G6", "I am in pain. Please help me."),
    ("G7", "Stop. Please stop."),
    ("G8", "I would like more, please."),
    ("G9", "Please wait a moment."),
    ("G10", "I need to use the bathroom."),
];

/// Template engine state
pub struct TemplateEngine {
    templates: HashMap<String, String>,
    custom_templates: HashMap<String, String>,
}

impl TemplateEngine {
    pub fn new() -> Self {
        let mut templates = HashMap::new();
        for (gesture_id, template) in TEMPLATES {
            templates.insert(gesture_id.to_string(), template.to_string());
        }
        
        Self {
            templates,
            custom_templates: HashMap::new(),
        }
    }
    
    /// Get the phrase for a gesture ID
    /// 
    /// Returns custom template if available, otherwise default template.
    pub fn get_phrase(&self, gesture_id: &str) -> Option<String> {
        // Check custom templates first
        if let Some(custom) = self.custom_templates.get(gesture_id) {
            return Some(custom.clone());
        }
        
        // Fall back to default templates
        self.templates.get(gesture_id).cloned()
    }
    
    /// Set a custom template for a gesture
    /// 
    /// This is used during correction - user can customize phrasing
    /// but it must still be a deterministic template (no LLM).
    pub fn set_custom_template(&mut self, gesture_id: &str, template: String) {
        self.custom_templates.insert(gesture_id.to_string(), template);
    }
    
    /// Clear all custom templates
    pub fn clear_custom_templates(&mut self) {
        self.custom_templates.clear();
    }
    
    /// Get all templates (for debugging/export)
    pub fn get_all_templates(&self) -> HashMap<String, String> {
        let mut all = self.templates.clone();
        all.extend(self.custom_templates.clone());
        all
    }
}

impl Default for TemplateEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_templates() {
        let engine = TemplateEngine::new();
        
        assert_eq!(engine.get_phrase("G1"), Some("I would like some water, please.".to_string()));
        assert_eq!(engine.get_phrase("G3"), Some("Yes.".to_string()));
        assert_eq!(engine.get_phrase("G10"), Some("I need to use the bathroom.".to_string()));
    }
    
    #[test]
    fn test_custom_template() {
        let mut engine = TemplateEngine::new();
        
        engine.set_custom_template("G1", "Water, please.".to_string());
        
        assert_eq!(engine.get_phrase("G1"), Some("Water, please.".to_string()));
        // Other templates unchanged
        assert_eq!(engine.get_phrase("G2"), Some("Could you help me, please?".to_string()));
    }
    
    #[test]
    fn test_clear_custom_templates() {
        let mut engine = TemplateEngine::new();
        
        engine.set_custom_template("G1", "Custom".to_string());
        engine.clear_custom_templates();
        
        assert_eq!(engine.get_phrase("G1"), Some("I would like some water, please.".to_string()));
    }
}
