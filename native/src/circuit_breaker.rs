// Circuit Breaker Module for Gestalt-MVP
// Phase 3: Mobile Bridging - Backpressure and degradation handling

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use log::{info, warn, error};

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CircuitState {
    /// Normal operation
    Closed,
    /// Failing, waiting to trip
    Opening,
    /// Open, rejecting requests
    Open,
    /// Half-open, testing recovery
    HalfOpen,
}

/// Health status for monitoring
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub circuit_state: CircuitState,
    pub failure_count: u32,
    pub success_count: u32,
    pub last_failure_time: Option<Instant>,
    pub consecutive_failures: u32,
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self {
            circuit_state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure_time: None,
            consecutive_failures: 0,
        }
    }
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening circuit
    pub failure_threshold: u32,
    /// Time to wait before trying again (half-open)
    pub reset_timeout: Duration,
    /// Timeout for individual operations
    pub operation_timeout: Duration,
    /// Number of successes in half-open to close circuit
    pub success_threshold: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            reset_timeout: Duration::from_secs(30),
            operation_timeout: Duration::from_millis(500),
            success_threshold: 3,
        }
    }
}

/// Circuit breaker for backpressure and graceful degradation
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<Mutex<CircuitState>>,
    health: Arc<Mutex<HealthStatus>>,
    half_open_successes: Arc<Mutex<u32>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker with default config
    pub fn new() -> Self {
        Self::with_config(CircuitBreakerConfig::default())
    }
    
    /// Create with custom configuration
    pub fn with_config(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(CircuitState::Closed)),
            health: Arc::new(Mutex::new(HealthStatus::default())),
            half_open_successes: Arc::new(Mutex::new(0)),
        }
    }
    
    /// Get current circuit state
    pub fn get_state(&self) -> CircuitState {
        *self.state.lock().unwrap()
    }
    
    /// Get health status
    pub fn get_health(&self) -> HealthStatus {
        self.health.lock().unwrap().clone()
    }
    
    /// Check if request should be allowed through
    pub fn allow_request(&self) -> bool {
        let mut state_guard = self.state.lock().unwrap();
        let mut health_guard = self.health.lock().unwrap();
        
        match *state_guard {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if reset timeout has elapsed
                if let Some(last_failure) = health_guard.last_failure_time {
                    if last_failure.elapsed() >= self.config.reset_timeout {
                        info!("Circuit breaker transitioning to HalfOpen");
                        *state_guard = CircuitState::HalfOpen;
                        *self.half_open_successes.lock().unwrap() = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitState::Opening => true,
            CircuitState::HalfOpen => true,
        }
    }
    
    /// Record a successful operation
    pub fn record_success(&self) {
        let mut state_guard = self.state.lock().unwrap();
        let mut health_guard = self.health.lock().unwrap();
        
        health_guard.success_count += 1;
        health_guard.consecutive_failures = 0;
        
        match *state_guard {
            CircuitState::HalfOpen => {
                let mut half_open_successes = self.half_open_successes.lock().unwrap();
                *half_open_successes += 1;
                
                if *half_open_successes >= self.config.success_threshold {
                    info!("Circuit breaker transitioning to Closed (recovered)");
                    *state_guard = CircuitState::Closed;
                }
            }
            CircuitState::Opening => {
                *state_guard = CircuitState::Closed;
            }
            _ => {}
        }
    }
    
    /// Record a failed operation
    pub fn record_failure(&self) {
        let mut state_guard = self.state.lock().unwrap();
        let mut health_guard = self.health.lock().unwrap();
        
        health_guard.failure_count += 1;
        health_guard.consecutive_failures += 1;
        health_guard.last_failure_time = Some(Instant::now());
        
        match *state_guard {
            CircuitState::Closed => {
                if health_guard.consecutive_failures >= self.config.failure_threshold {
                    warn!(
                        "Circuit breaker tripping: {} consecutive failures",
                        health_guard.consecutive_failures
                    );
                    *state_guard = CircuitState::Open;
                } else {
                    *state_guard = CircuitState::Opening;
                }
            }
            CircuitState::Opening => {
                *state_guard = CircuitState::Open;
            }
            CircuitState::HalfOpen => {
                warn!("Circuit breaker re-tripping from HalfOpen");
                *state_guard = CircuitState::Open;
            }
            CircuitState::Open => {}
        }
    }
    
    /// Execute an operation with circuit breaker protection
    pub fn execute<T, F>(&self, operation: F) -> Result<T, String>
    where
        F: FnOnce() -> Result<T, String>,
    {
        if !self.allow_request() {
            return Err("Circuit breaker is open".to_string());
        }
        
        let result = operation();
        
        match &result {
            Ok(_) => self.record_success(),
            Err(_) => self.record_failure(),
        }
        
        result
    }
    
    /// Reset the circuit breaker to initial state
    pub fn reset(&self) {
        *self.state.lock().unwrap() = CircuitState::Closed;
        *self.health.lock().unwrap() = HealthStatus::default();
        *self.half_open_successes.lock().unwrap() = 0;
        info!("Circuit breaker manually reset");
    }
    
    /// Force circuit to open (emergency stop)
    pub fn force_open(&self) {
        *self.state.lock().unwrap() = CircuitState::Open;
        *self.health.lock().unwrap() = HealthStatus {
            circuit_state: CircuitState::Open,
            ..Default::default()
        };
        warn!("Circuit breaker forced open");
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}

/// Graceful degradation strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DegradationLevel {
    /// Full functionality
    Full,
    /// Reduced frame rate
    ReducedFrameRate,
    /// Skip non-critical processing
    SkipNonCritical,
    /// Only essential operations
    EssentialOnly,
    /// Complete shutdown
    Shutdown,
}

/// Degradation manager for adaptive performance
pub struct DegradationManager {
    circuit_breaker: CircuitBreaker,
    current_level: Arc<Mutex<DegradationLevel>>,
    frame_skip_count: Arc<Mutex<u32>>,
    target_frame_interval: Arc<Mutex<u32>>,
}

impl DegradationManager {
    /// Create a new degradation manager
    pub fn new() -> Self {
        Self {
            circuit_breaker: CircuitBreaker::new(),
            current_level: Arc::new(Mutex::new(DegradationLevel::Full)),
            frame_skip_count: Arc::new(Mutex::new(0)),
            target_frame_interval: Arc::new(Mutex::new(1)), // Process every frame
        }
    }
    
    /// Get current degradation level
    pub fn get_level(&self) -> DegradationLevel {
        *self.current_level.lock().unwrap()
    }
    
    /// Get circuit breaker reference
    pub fn circuit_breaker(&self) -> &CircuitBreaker {
        &self.circuit_breaker
    }
    
    /// Update degradation level based on system load
    pub fn update_level(&self, latency_ms: u32, fps: f32) {
        let mut level_guard = self.current_level.lock().unwrap();
        let mut skip_guard = self.frame_skip_count.lock().unwrap();
        let mut interval_guard = self.target_frame_interval.lock().unwrap();
        
        let new_level = if latency_ms > 500 || fps < 5.0 {
            *skip_guard = 4; // Process 1 in 5 frames
            *interval_guard = 5;
            DegradationLevel::ReducedFrameRate
        } else if latency_ms > 300 || fps < 10.0 {
            *skip_guard = 2; // Process 1 in 3 frames
            *interval_guard = 3;
            DegradationLevel::SkipNonCritical
        } else if latency_ms > 200 || fps < 15.0 {
            *skip_guard = 1; // Process 2 in 3 frames
            *interval_guard = 2;
            DegradationLevel::EssentialOnly
        } else {
            *skip_guard = 0;
            *interval_guard = 1;
            DegradationLevel::Full
        };
        
        if new_level != *level_guard {
            info!(
                "Degradation level changed: {:?} → {:?}",
                level_guard, new_level
            );
            *level_guard = new_level;
        }
    }
    
    /// Check if current frame should be processed
    pub fn should_process_frame(&self, frame_number: u64) -> bool {
        let skip_count = *self.frame_skip_count.lock().unwrap();
        if skip_count == 0 {
            return true;
        }
        
        let interval = *self.target_frame_interval.lock().unwrap();
        frame_number % (interval as u64) == 0
    }
    
    /// Get recommended action based on current state
    pub fn get_action(&self) -> &'static str {
        match self.get_level() {
            DegradationLevel::Full => "Process all frames normally",
            DegradationLevel::ReducedFrameRate => "Skip frames, reduce processing",
            DegradationLevel::SkipNonCritical => "Skip guards and non-critical checks",
            DegradationLevel::EssentialOnly => "Only DTW matching, no confidence",
            DegradationLevel::Shutdown => "Pause all processing",
        }
    }
}

impl Default for DegradationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_circuit_breaker_basic() {
        let cb = CircuitBreaker::new();
        assert_eq!(cb.get_state(), CircuitState::Closed);
        
        // Record some successes
        cb.record_success();
        cb.record_success();
        assert_eq!(cb.get_state(), CircuitState::Closed);
    }
    
    #[test]
    fn test_circuit_breaker_trips() {
        let cb = CircuitBreaker::with_config(CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        });
        
        // Trip the circuit
        for _ in 0..3 {
            cb.record_failure();
        }
        
        assert_eq!(cb.get_state(), CircuitState::Open);
        assert!(!cb.allow_request());
    }
    
    #[test]
    fn test_degradation_manager() {
        let dm = DegradationManager::new();
        assert_eq!(dm.get_level(), DegradationLevel::Full);
        
        // Simulate high latency
        dm.update_level(600, 3.0);
        assert_eq!(dm.get_level(), DegradationLevel::ReducedFrameRate);
        
        // Simulate normal conditions
        dm.update_level(50, 30.0);
        assert_eq!(dm.get_level(), DegradationLevel::Full);
    }
}
