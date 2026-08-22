"""
Segmentation module for Gestalt-MVP hand trajectory data.

Implements the frozen segmentation logic from Section 5:
- Motion Start: wrist velocity energy > τ_start for 3 consecutive frames
- Motion End: wrist velocity energy < τ_rest for ≥ 300ms (9 frames @30fps)
- Duration bounds: discard segments < 0.3s or > 2.5s
- All gestures are PURELY DYNAMIC (motion + return to rest)
- NO static holds
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np


# Default thresholds (tuned on Session 3, frozen thereafter)
DEFAULT_TAU_START = 0.0025  # Velocity energy threshold for motion start
DEFAULT_TAU_REST = 0.001    # Velocity energy threshold for rest
DEFAULT_FPS = 30.0
MIN_DURATION_S = 0.3
MAX_DURATION_S = 2.5
FRAMES_FOR_START = 3        # Consecutive frames above tau_start
FRAMES_FOR_REST = 9         # Consecutive frames below tau_rest (300ms @ 30fps)


def calculate_wrist_velocity_energy(landmarks_prev: np.ndarray,
                                     landmarks_curr: np.ndarray,
                                     dt: float) -> float:
    """
    Calculate squared L2 norm of wrist velocity between two frames.
    
    Args:
        landmarks_prev: Previous frame landmarks (21, 3)
        landmarks_curr: Current frame landmarks (21, 3)
        dt: Time delta in seconds
        
    Returns:
        Velocity energy (squared L2 norm)
    """
    wrist_prev = landmarks_prev[0, :]
    wrist_curr = landmarks_curr[0, :]
    velocity = (wrist_curr - wrist_prev) / dt
    return float(np.dot(velocity, velocity))


def compute_velocity_energies(frames: List[Dict[str, Any]], 
                               fps: float = DEFAULT_FPS) -> List[float]:
    """
    Compute wrist velocity energy for each frame transition.
    
    Args:
        frames: List of frame dicts with 'landmarks' key
        fps: Frames per second
        
    Returns:
        List of velocity energies (length = len(frames) - 1)
    """
    if len(frames) < 2:
        return []
    
    dt = 1.0 / fps
    energies = []
    
    for i in range(1, len(frames)):
        lm_prev = np.array(frames[i-1]['landmarks'], dtype=np.float64)
        lm_curr = np.array(frames[i]['landmarks'], dtype=np.float64)
        energy = calculate_wrist_velocity_energy(lm_prev, lm_curr, dt)
        energies.append(energy)
    
    return energies


def detect_motion_start(energies: List[float], 
                        tau_start: float = DEFAULT_TAU_START,
                        frames_required: int = FRAMES_FOR_START) -> Optional[int]:
    """
    Detect the start of motion based on velocity energy threshold.
    
    Motion starts when velocity energy exceeds tau_start for 
    frames_required consecutive frames.
    
    Args:
        energies: List of velocity energies
        tau_start: Threshold for motion detection
        frames_required: Number of consecutive frames above threshold
        
    Returns:
        Index of first frame where motion starts, or None if not detected
    """
    if len(energies) < frames_required:
        return None
    
    consecutive_count = 0
    
    for i, energy in enumerate(energies):
        if energy > tau_start:
            consecutive_count += 1
            if consecutive_count >= frames_required:
                # Return the index of the first frame in the sequence
                return i - frames_required + 1
        else:
            consecutive_count = 0
    
    return None


def detect_motion_end(energies: List[float],
                      start_idx: int,
                      tau_rest: float = DEFAULT_TAU_REST,
                      frames_required: int = FRAMES_FOR_REST) -> Optional[int]:
    """
    Detect the end of motion based on velocity energy dropping below threshold.
    
    Motion ends when velocity energy stays below tau_rest for 
    frames_required consecutive frames (300ms).
    
    Args:
        energies: List of velocity energies
        start_idx: Index where motion started
        tau_rest: Threshold for rest detection
        frames_required: Number of consecutive frames below threshold
        
    Returns:
        Index where motion ends (first frame of rest sequence), or None
    """
    if start_idx >= len(energies):
        return None
    
    consecutive_count = 0
    
    for i in range(start_idx, len(energies)):
        if energies[i] < tau_rest:
            consecutive_count += 1
            if consecutive_count >= frames_required:
                # Return the index of the last moving frame (before rest)
                return i - frames_required + 1
        else:
            consecutive_count = 0
    
    return None


def segment_trajectory(frames: List[Dict[str, Any]],
                       tau_start: float = DEFAULT_TAU_START,
                       tau_rest: float = DEFAULT_TAU_REST,
                       fps: float = DEFAULT_FPS,
                       min_duration: float = MIN_DURATION_S,
                       max_duration: float = MAX_DURATION_S) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Segment a continuous stream of frames into a single gesture.
    
    Implements REST → MOTION → REST boundary detection.
    
    Args:
        frames: List of frame dicts with 't' and 'landmarks' keys
        tau_start: Motion start threshold
        tau_rest: Rest detection threshold
        fps: Frames per second
        min_duration: Minimum valid gesture duration in seconds
        max_duration: Maximum valid gesture duration in seconds
        
    Returns:
        Tuple of (segmented_frames, error_message):
        - segmented_frames: List of frames in the gesture, or None if invalid
        - error_message: Error description, or None if successful
    """
    if len(frames) < 2:
        return None, "Insufficient frames for segmentation"
    
    # Compute velocity energies
    energies = compute_velocity_energies(frames, fps)
    
    if len(energies) < FRAMES_FOR_START + FRAMES_FOR_REST:
        return None, "Trajectory too short for reliable segmentation"
    
    # Detect motion start
    start_idx = detect_motion_start(energies, tau_start, FRAMES_FOR_START)
    
    if start_idx is None:
        return None, "No motion start detected"
    
    # Detect motion end
    end_idx = detect_motion_end(energies, start_idx, tau_rest, FRAMES_FOR_REST)
    
    if end_idx is None:
        return None, "No motion end detected (gesture incomplete)"
    
    # Extract segment (include frames from start to end inclusive)
    segment = frames[start_idx:end_idx + 1]
    
    # Validate duration
    duration = segment[-1]['t'] - segment[0]['t']
    
    if duration < min_duration:
        return None, f"Gesture too short ({duration:.2f}s < {min_duration}s)"
    
    if duration > max_duration:
        return None, f"Gesture too long ({duration:.2f}s > {max_duration}s)"
    
    return segment, None


def find_segments_in_stream(frames: List[Dict[str, Any]],
                            tau_start: float = DEFAULT_TAU_START,
                            tau_rest: float = DEFAULT_TAU_REST,
                            fps: float = DEFAULT_FPS,
                            min_duration: float = MIN_DURATION_S,
                            max_duration: float = MAX_DURATION_S) -> List[Tuple[int, int, List[Dict[str, Any]]]]:
    """
    Find all valid gesture segments in a continuous stream.
    
    Useful for processing longer recordings that may contain multiple gestures.
    
    Args:
        frames: List of frame dicts
        tau_start: Motion start threshold
        tau_rest: Rest detection threshold
        fps: Frames per second
        min_duration: Minimum valid gesture duration
        max_duration: Maximum valid gesture duration
        
    Returns:
        List of tuples (start_idx, end_idx, segment_frames)
    """
    segments = []
    current_pos = 0
    
    while current_pos < len(frames) - FRAMES_FOR_START:
        # Try to find a segment starting from current position
        remaining_frames = frames[current_pos:]
        segment, error = segment_trajectory(
            remaining_frames, tau_start, tau_rest, fps, min_duration, max_duration
        )
        
        if segment is not None:
            # Found a valid segment
            end_pos = current_pos + len(segment) - 1
            segments.append((current_pos, end_pos, segment))
            # Move past this segment to find next one
            current_pos = end_pos + FRAMES_FOR_REST
        else:
            # No segment found; advance by one frame
            current_pos += 1
    
    return segments


def validate_segment_quality(segment: List[Dict[str, Any]],
                             fps: float = DEFAULT_FPS) -> Dict[str, float]:
    """
    Compute quality metrics for a segmented gesture.
    
    Args:
        segment: List of frames in the gesture
        fps: Frames per second
        
    Returns:
        Dictionary with quality metrics
    """
    if len(segment) < 2:
        return {'valid': False, 'reason': 'too_few_frames'}
    
    duration = segment[-1]['t'] - segment[0]['t']
    frame_count = len(segment)
    energies = compute_velocity_energies(segment, fps)
    
    # Check for return-to-rest pattern
    if len(energies) >= 3:
        initial_energy = np.mean(energies[:3])
        final_energy = np.mean(energies[-3:])
        peak_energy = np.max(energies)
        
        # Should have low energy at start and end, high in middle
        has_return_to_rest = (initial_energy < DEFAULT_TAU_START * 2 and 
                              final_energy < DEFAULT_TAU_START * 2 and
                              peak_energy > DEFAULT_TAU_START)
    else:
        has_return_to_rest = False
    
    return {
        'valid': True,
        'duration_s': duration,
        'frame_count': frame_count,
        'avg_energy': float(np.mean(energies)) if energies else 0.0,
        'peak_energy': float(np.max(energies)) if energies else 0.0,
        'has_return_to_rest': has_return_to_rest
    }
