//! DTW Matcher Module for Gestalt-MVP
//! 
//! Implements weighted Dynamic Time Warping with Sakoe-Chiba band.
//! Supports multi-exemplar matching and margin-based open-set gating.

use crate::graph::{Trajectory, Frame};
use crate::message_bus::{MatchResult, GestureId};

/// Landmark weights for DTW cost function
/// Fingertips (indices 4, 8, 12, 16, 20): weight = 2.0
/// All other landmarks: weight = 1.0
const LANDMARK_WEIGHTS: [f64; 21] = [
    1.0, // 0: Wrist
    1.0, // 1: Thumb CMC
    1.0, // 2: Thumb MCP
    1.0, // 3: Thumb IP
    2.0, // 4: Thumb TIP (fingertip)
    1.0, // 5: Index MCP
    1.0, // 6: Index PIP
    1.0, // 7: Index DIP
    2.0, // 8: Index TIP (fingertip)
    1.0, // 9: Middle MCP
    1.0, // 10: Middle PIP
    1.0, // 11: Middle DIP
    2.0, // 12: Middle TIP (fingertip)
    1.0, // 13: Ring MCP
    1.0, // 14: Ring PIP
    1.0, // 15: Ring DIP
    2.0, // 16: Ring TIP (fingertip)
    1.0, // 17: Pinky MCP
    1.0, // 18: Pinky PIP
    1.0, // 19: Pinky DIP
    2.0, // 20: Pinky TIP (fingertip)
];

/// Weighted Euclidean distance between two frames
fn weighted_frame_distance(frame_a: &Frame, frame_b: &Frame) -> f64 {
    let mut sum_sq = 0.0;
    
    for (i, (lm_a, lm_b)) in frame_a.landmarks.iter()
        .zip(frame_b.landmarks.iter())
        .enumerate()
    {
        let dx = (lm_a[0] - lm_b[0]) as f64;
        let dy = (lm_a[1] - lm_b[1]) as f64;
        let dz = (lm_a[2] - lm_b[2]) as f64;
        
        let dist_sq = dx * dx + dy * dy + dz * dz;
        let weight = LANDMARK_WEIGHTS[i];
        
        sum_sq += weight * dist_sq;
    }
    
    sum_sq.sqrt()
}

/// Calculate Sakoe-Chiba band radius
/// r = max(3, floor(0.20 × max(L_x, L_y)))
fn sakoe_chiba_band(len_x: usize, len_y: usize) -> usize {
    let max_len = len_x.max(len_y);
    let band = (0.20 * max_len as f64).floor() as usize;
    band.max(3)
}

/// Dynamic Time Warping with Sakoe-Chiba band constraint
/// 
/// Returns the accumulated distance between two trajectories.
pub fn dtw_distance(traj_a: &Trajectory, traj_b: &Trajectory) -> f64 {
    let n = traj_a.frames.len();
    let m = traj_b.frames.len();
    
    if n == 0 || m == 0 {
        return f64::INFINITY;
    }
    
    let band = sakoe_chiba_band(n, m);
    
    // Initialize DP matrix with infinity
    let mut dp = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dp[0][0] = 0.0;
    
    // Fill DP matrix with band constraint
    for i in 1..=n {
        let j_start = 1.max((i as isize - band as isize) as usize);
        let j_end = m.min(i + band);
        
        for j in j_start..=j_end {
            let cost = weighted_frame_distance(&traj_a.frames[i - 1], &traj_b.frames[j - 1]);
            
            dp[i][j] = cost + dp[i - 1][j - 1]
                .min(dp[i - 1][j])
                .min(dp[i][j - 1]);
        }
    }
    
    dp[n][m]
}

/// Match a query trajectory against all exemplars of all gestures
/// 
/// Returns the best match (D1), second-best match (D2), and computed margin.
pub fn match_trajectory(
    query: &Trajectory,
    gesture_library: &[(GestureId, Vec<Trajectory>)],
) -> MatchResult {
    let mut distances: Vec<(GestureId, f64)> = Vec::new();
    
    // Find minimum distance to each gesture class
    for (gesture_id, exemplars) in gesture_library {
        if exemplars.is_empty() {
            continue;
        }
        
        let min_dist = exemplars
            .iter()
            .map(|exemplar| dtw_distance(query, exemplar))
            .fold(f64::INFINITY, f64::min);
        
        distances.push((gesture_id.clone(), min_dist));
    }
    
    if distances.is_empty() {
        return MatchResult::new(None, f64::INFINITY, f64::INFINITY, 0);
    }
    
    // Sort by distance
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    
    let best_gesture = distances[0].0.clone();
    let d1 = distances[0].1;
    let d2 = if distances.len() > 1 { distances[1].1 } else { f64::INFINITY };
    
    // Count total exemplars
    let total_exemplars: usize = gesture_library.iter().map(|(_, ex)| ex.len()).sum();
    
    MatchResult::new(Some(best_gesture), d1, d2, total_exemplars)
}

/// Batch matcher for efficiency (brute-force nearest neighbor)
pub struct DtwMatcher {
    gesture_library: Vec<(GestureId, Vec<Trajectory>)>,
}

impl DtwMatcher {
    pub fn new() -> Self {
        Self {
            gesture_library: Vec::new(),
        }
    }
    
    /// Add exemplars for a gesture class
    pub fn add_gesture(&mut self, gesture_id: &str, exemplars: Vec<Trajectory>) {
        self.gesture_library.push((gesture_id.to_string(), exemplars));
    }
    
    /// Get total number of enrolled exemplars
    pub fn exemplar_count(&self) -> usize {
        self.gesture_library.iter().map(|(_, ex)| ex.len()).sum()
    }
    
    /// Match a query trajectory
    pub fn match_query(&self, query: &Trajectory) -> MatchResult {
        match_trajectory(query, &self.gesture_library)
    }
    
    /// Clear all enrolled gestures
    pub fn clear(&mut self) {
        self.gesture_library.clear();
    }
}

impl Default for DtwMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_frame(t: f64, offset: f32) -> Frame {
        let mut landmarks = Vec::with_capacity(21);
        for i in 0..21 {
            landmarks.push([
                (i % 5) as f32 / 10.0 + offset,
                (i / 5) as f32 / 10.0 + offset,
                0.5 + offset,
            ]);
        }
        Frame { t, landmarks }
    }

    fn create_test_trajectory(frame_count: usize, offset: f32) -> Trajectory {
        let mut frames = Vec::with_capacity(frame_count);
        for i in 0..frame_count {
            frames.push(create_test_frame(i as f64 * 0.033, offset + i as f32 * 0.01));
        }
        Trajectory {
            frames,
            duration_s: frame_count as f64 * 0.033,
            frame_count,
        }
    }

    #[test]
    fn test_weighted_distance() {
        let frame1 = create_test_frame(0.0, 0.0);
        let frame2 = create_test_frame(0.0, 0.0);
        
        // Identical frames should have zero distance
        let dist = weighted_frame_distance(&frame1, &frame2);
        assert!(dist < 1e-6);
    }

    #[test]
    fn test_dtw_identical_trajectories() {
        let traj = create_test_trajectory(10, 0.0);
        let dist = dtw_distance(&traj, &traj);
        
        // Distance to self should be very small (near zero)
        assert!(dist < 1e-5);
    }

    #[test]
    fn test_dtw_different_trajectories() {
        let traj1 = create_test_trajectory(10, 0.0);
        let traj2 = create_test_trajectory(10, 0.5); // Different offset
        
        let dist_same = dtw_distance(&traj1, &traj1);
        let dist_diff = dtw_distance(&traj1, &traj2);
        
        // Different trajectories should have larger distance
        assert!(dist_diff > dist_same);
    }

    #[test]
    fn test_matcher_with_multiple_gestures() {
        let mut matcher = DtwMatcher::new();
        
        // Add two gesture classes
        let g1_exemplars = vec![create_test_trajectory(10, 0.0)];
        let g2_exemplars = vec![create_test_trajectory(10, 0.5)];
        
        matcher.add_gesture("G1", g1_exemplars);
        matcher.add_gesture("G2", g2_exemplars);
        
        // Query similar to G1
        let query = create_test_trajectory(10, 0.01);
        let result = matcher.match_query(&query);
        
        assert_eq!(result.gesture_id, Some("G1".to_string()));
        assert!(result.distance_to_best < result.distance_to_second);
        assert!(result.margin > 0.0);
    }

    #[test]
    fn test_sakoe_chiba_band() {
        assert_eq!(sakoe_chiba_band(10, 10), 3); // 0.2 * 10 = 2, max(3, 2) = 3
        assert_eq!(sakoe_chiba_band(50, 50), 10); // 0.2 * 50 = 10
        assert_eq!(sakoe_chiba_band(100, 100), 20); // 0.2 * 100 = 20
    }
}
