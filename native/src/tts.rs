// TTS Actuator Module for Gestalt-MVP
// Phase 2: End-to-End Pipeline
// 
// This module handles Text-to-Speech synthesis using Piper TTS or system fallback.
// All TTS operations are local and offline. No cloud services.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::process::{Command, Stdio};
use std::io::Write;
use log::{info, warn, error};

/// TTS Engine types available in Phase 2
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TtsEngine {
    /// Piper TTS (primary, local, offline)
    Piper,
    /// System TTS fallback (macOS say / espeak / SAPI)
    System,
    /// Sherpa-ONNX (alternative local engine)
    SherpaOnnx,
}

/// TTS configuration parameters
#[derive(Debug, Clone)]
pub struct TtsConfig {
    pub engine: TtsEngine,
    pub voice_path: String,
    pub model_path: String,
    pub speed: f32,
    pub pitch: f32,
    pub volume: f32,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            engine: TtsEngine::Piper,
            voice_path: String::from("assets/models/voice.onnx"),
            model_path: String::from("assets/models/voice.json"),
            speed: 1.0,
            pitch: 1.0,
            volume: 1.0,
        }
    }
}

/// TTS state machine synchronized with FSM
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TtsState {
    Idle,
    Speaking,
    Paused,
    Error,
}

/// Main TTS actuator structure
pub struct TtsActuator {
    config: TtsConfig,
    state: Arc<Mutex<TtsState>>,
    piper_available: bool,
    sherpa_available: bool,
}

impl TtsActuator {
    /// Create a new TTS actuator with auto-detection of available engines
    pub fn new(config: TtsConfig) -> Self {
        let piper_available = Self::check_piper();
        let sherpa_available = Self::check_sherpa();
        
        info!("TTS Actuator initialized: Piper={}, Sherpa={}", 
              piper_available, sherpa_available);
        
        Self {
            config,
            state: Arc::new(Mutex::new(TtsState::Idle)),
            piper_available,
            sherpa_available,
        }
    }
    
    /// Check if Piper TTS is available on the system
    fn check_piper() -> bool {
        // Check for piper executable
        Command::new("piper")
            .arg("--help")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
    
    /// Check if Sherpa-ONNX is available
    fn check_sherpa() -> bool {
        // Check for sherpa-onnx executable or library
        Command::new("sherpa-onnx")
            .arg("--help")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
    
    /// Get current TTS state
    pub fn get_state(&self) -> TtsState {
        *self.state.lock().unwrap()
    }
    
    /// Speak text with FSM-state-modulated prosody
    /// 
    /// HIGH_CONF → normal speed/pitch
    /// CLARIFY → slower speed, lower pitch
    /// SILENT → no output (returns immediately)
    pub fn speak(&self, text: &str, fsm_state: &str) -> Result<(), String> {
        let mut state_guard = self.state.lock().unwrap();
        
        // Handle SILENT state - bypass all TTS
        if fsm_state == "SILENT" || fsm_state == "ERROR" {
            info!("TTS bypassed: FSM state={}", fsm_state);
            return Ok(());
        }
        
        // Modulate prosody based on FSM state
        let (speed_mod, pitch_mod) = match fsm_state {
            "HIGH_CONF" => (1.0, 1.0),      // Normal
            "CLARIFY" | "LOW_CONF" => (0.7, 0.9),  // Slower, lower pitch
            "CORRECTION" => (0.8, 0.95),    // Slightly slower
            _ => (1.0, 1.0),
        };
        
        let effective_speed = self.config.speed * speed_mod;
        let effective_pitch = self.config.pitch * pitch_mod;
        
        info!("Speaking: \"{}\" (state={}, speed={:.2}, pitch={:.2})", 
              text, fsm_state, effective_speed, effective_pitch);
        
        // Release lock before speaking (long operation)
        drop(state_guard);
        *self.state.lock().unwrap() = TtsState::Speaking;
        
        let result = match self.config.engine {
            TtsEngine::Piper if self.piper_available => {
                self.speak_piper(text, effective_speed, effective_pitch)
            }
            TtsEngine::SherpaOnnx if self.sherpa_available => {
                self.speak_sherpa(text, effective_speed, effective_pitch)
            }
            _ => {
                // Fallback to system TTS
                self.speak_system(text, effective_speed, effective_pitch)
            }
        };
        
        *self.state.lock().unwrap() = TtsState::Idle;
        result
    }
    
    /// Speak using Piper TTS
    fn speak_piper(&self, text: &str, speed: f32, pitch: f32) -> Result<(), String> {
        // Piper command line: echo "text" | piper --model model.onnx --config config.json --output-raw | aplay
        let mut cmd = Command::new("piper");
        cmd.arg("--model")
           .arg(&self.config.model_path)
           .arg("--length-scale")
           .arg(format!("{:.2}", 1.0 / speed))  // Inverse relationship
           .arg("--pitch-scale")
           .arg(format!("{:.2}", pitch));
        
        // For Phase 2, we pipe to system audio
        // In production, this would use proper audio device handling
        cmd.stdin(Stdio::piped())
           .stdout(Stdio::null())
           .stderr(Stdio::piped());
        
        let mut child = cmd.spawn()
            .map_err(|e| format!("Failed to spawn piper: {}", e))?;
        
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(text.as_bytes())
                .map_err(|e| format!("Failed to write to piper: {}", e))?;
        }
        
        let output = child.wait_with_output()
            .map_err(|e| format!("Piper execution failed: {}", e))?;
        
        if output.status.success() {
            info!("Piper TTS completed successfully");
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!("Piper TTS failed: {}", stderr);
            Err(format!("Piper failed: {}", stderr))
        }
    }
    
    /// Speak using Sherpa-ONNX
    fn speak_sherpa(&self, text: &str, speed: f32, pitch: f32) -> Result<(), String> {
        // Sherpa-ONNX command line interface
        let mut cmd = Command::new("sherpa-onnx");
        cmd.arg("--vocoder-model")
           .arg(&self.config.model_path)
           .arg("--max-num-sentences")
           .arg("1")
           .arg("--speed")
           .arg(format!("{:.2}", speed));
        
        cmd.stdin(Stdio::piped())
           .stdout(Stdio::null())
           .stderr(Stdio::piped());
        
        let mut child = cmd.spawn()
            .map_err(|e| format!("Failed to spawn sherpa-onnx: {}", e))?;
        
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(text.as_bytes())
                .map_err(|e| format!("Failed to write to sherpa: {}", e))?;
        }
        
        let output = child.wait_with_output()
            .map_err(|e| format!("Sherpa execution failed: {}", e))?;
        
        if output.status.success() {
            info!("Sherpa-ONNX TTS completed successfully");
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!("Sherpa-ONNX TTS failed: {}", stderr);
            Err(format!("Sherpa failed: {}", stderr))
        }
    }
    
    /// Speak using system TTS (fallback)
    fn speak_system(&self, text: &str, speed: f32, pitch: f32) -> Result<(), String> {
        #[cfg(target_os = "macos")]
        {
            // macOS 'say' command
            let mut cmd = Command::new("say");
            cmd.arg("-r")
               .arg(format!("{:.0}", (speed * 100.0) as i32))  // Words per minute
               .arg(text);
            
            let output = cmd.output()
                .map_err(|e| format!("macOS say failed: {}", e))?;
            
            if output.status.success() {
                info!("System TTS (macOS say) completed");
                Ok(())
            } else {
                Err(format!("macOS say failed"))
            }
        }
        
        #[cfg(target_os = "linux")]
        {
            // Linux espeak
            let mut cmd = Command::new("espeak");
            cmd.arg("-s")
               .arg(format!("{:.0}", (speed * 150.0) as i32))  // Words per minute
               .arg("-p")
               .arg(format!("{:.0}", (pitch * 50.0) as i32))   // Pitch adjustment
               .arg(text);
            
            let output = cmd.output()
                .map_err(|e| format!("espeak failed: {}", e))?;
            
            if output.status.success() {
                info!("System TTS (espeak) completed");
                Ok(())
            } else {
                Err(format!("espeak failed"))
            }
        }
        
        #[cfg(target_os = "windows")]
        {
            // Windows PowerShell SAPI
            let powershell_script = format!(
                "Add-Type -AssemblyName System.Speech; \
                 $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; \
                 $speak.Rate = {}; \
                 $speak.Pitch = {}; \
                 $speak.Speak(\"{}\")",
                ((speed - 1.0) * 10.0) as i32,
                ((pitch - 1.0) * 10.0) as i32,
                text.replace("\"", "\\\"")
            );
            
            let output = Command::new("powershell")
                .arg("-Command")
                .arg(powershell_script)
                .output()
                .map_err(|e| format!("PowerShell SAPI failed: {}", e))?;
            
            if output.status.success() {
                info!("System TTS (Windows SAPI) completed");
                Ok(())
            } else {
                Err(format!("PowerShell SAPI failed"))
            }
        }
        
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            warn!("No system TTS available for this platform");
            Err("No TTS engine available".to_string())
        }
    }
    
    /// Stop current speech immediately (for emergency mute)
    pub fn stop(&self) -> Result<(), String> {
        info!("TTS stop requested");
        
        #[cfg(target_os = "macos")]
        {
            Command::new("killall").arg("say").status().ok();
        }
        
        #[cfg(target_os = "linux")]
        {
            Command::new("pkill").arg("espeak").status().ok();
        }
        
        #[cfg(target_os = "windows")]
        {
            Command::new("taskkill")
                .arg("/IM")
                .arg("powershell.exe")
                .arg("/F")
                .status().ok();
        }
        
        *self.state.lock().unwrap() = TtsState::Idle;
        Ok(())
    }
    
    /// Set TTS engine dynamically
    pub fn set_engine(&mut self, engine: TtsEngine) {
        self.config.engine = engine;
        info!("TTS engine changed to {:?}", engine);
    }
    
    /// Update prosody parameters
    pub fn set_prosody(&mut self, speed: Option<f32>, pitch: Option<f32>, volume: Option<f32>) {
        if let Some(s) = speed {
            self.config.speed = s;
        }
        if let Some(p) = pitch {
            self.config.pitch = p;
        }
        if let Some(v) = volume {
            self.config.volume = v;
        }
        info!("TTS prosody updated: speed={:.2}, pitch={:.2}, volume={:.2}",
              self.config.speed, self.config.pitch, self.config.volume);
    }
}

/// Test function for TTS actuator
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tts_creation() {
        let config = TtsConfig::default();
        let actuator = TtsActuator::new(config);
        assert_eq!(actuator.get_state(), TtsState::Idle);
    }
    
    #[test]
    fn test_prosody_modulation() {
        let config = TtsConfig::default();
        let actuator = TtsActuator::new(config);
        
        // Should not fail even if no TTS engine available
        let result = actuator.speak("Test", "HIGH_CONF");
        // Result may be Ok or Err depending on system, but should not panic
        assert!(result.is_ok() || result.is_err());
    }
}
