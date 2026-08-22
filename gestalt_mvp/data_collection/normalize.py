"""
Normalization module for Gestalt-MVP hand trajectory data.

Implements the frozen normalization math from Section 4:
1. Translation: p'_i = p_i - p_wrist
2. Scale: s = ||p'_MCP_middle||_2, p''_i = p'_i / s
3. Z-Rotation (azimuth alignment only): θ = atan2(p''_MCP_middle.y, p''_MCP_middle.x)
   Apply R(-θ) to X and Y of all landmarks. Z-coordinates preserved.

Landmark indices (MediaPipe Hands):
- 0: Wrist
- 9: MCP of middle finger
"""

import numpy as np
from typing import List, Dict, Any


def translate_to_wrist(landmarks: np.ndarray) -> np.ndarray:
    """
    Translate all landmarks so wrist (index 0) is at origin.
    
    Args:
        landmarks: Array of shape (21, 3) with [x, y, z] coordinates
        
    Returns:
        Translated landmarks of shape (21, 3)
    """
    wrist = landmarks[0:1, :]  # Keep dimensions (1, 3)
    return landmarks - wrist


def scale_by_mcp_middle(landmarks: np.ndarray) -> np.ndarray:
    """
    Scale landmarks by the distance from wrist to MCP of middle finger.
    
    Args:
        landmarks: Array of shape (21, 3), already translated to wrist origin
        
    Returns:
        Scaled landmarks of shape (21, 3)
    """
    mcp_middle = landmarks[9, :]  # Index 9 is MCP of middle finger
    scale = np.linalg.norm(mcp_middle[:2])  # Use only x, y for scale
    
    if scale < 1e-6:
        # Avoid division by zero; return as-is if scale is too small
        return landmarks
    
    return landmarks / scale


def rotate_z_azimuth(landmarks: np.ndarray) -> np.ndarray:
    """
    Rotate landmarks around Z-axis to align azimuth.
    
    Only rotates X and Y coordinates. Z (depth) is preserved.
    
    Args:
        landmarks: Array of shape (21, 3), already translated and scaled
        
    Returns:
        Rotated landmarks of shape (21, 3)
    """
    mcp_middle = landmarks[9, :]
    
    # Calculate azimuth angle in XY plane
    theta = np.arctan2(mcp_middle[1], mcp_middle[0])
    
    # Build 2D rotation matrix for -theta (align to positive X axis)
    cos_theta = np.cos(-theta)
    sin_theta = np.sin(-theta)
    
    # Apply rotation to X and Y only
    rotated = landmarks.copy()
    rotated[:, 0] = landmarks[:, 0] * cos_theta - landmarks[:, 1] * sin_theta
    rotated[:, 1] = landmarks[:, 0] * sin_theta + landmarks[:, 1] * cos_theta
    # Z coordinate is preserved
    
    return rotated


def normalize_frame(landmarks: List[List[float]]) -> List[List[float]]:
    """
    Apply full normalization pipeline to a single frame.
    
    Pipeline:
    1. Translation to wrist
    2. Scale by MCP-middle distance
    3. Z-rotation for azimuth alignment
    
    Args:
        landmarks: List of 21 [x, y, z] coordinates
        
    Returns:
        Normalized list of 21 [x, y, z] coordinates
    """
    # Convert to numpy array
    lm_array = np.array(landmarks, dtype=np.float64)
    
    # Step 1: Translation
    translated = translate_to_wrist(lm_array)
    
    # Step 2: Scaling
    scaled = scale_by_mcp_middle(translated)
    
    # Step 3: Z-rotation
    normalized = rotate_z_azimuth(scaled)
    
    return normalized.tolist()


def normalize_trajectory(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply normalization to all frames in a trajectory.
    
    Args:
        frames: List of frame dicts with 't' and 'landmarks' keys
        
    Returns:
        List of normalized frame dicts
    """
    normalized_frames = []
    
    for frame in frames:
        normalized_landmarks = normalize_frame(frame['landmarks'])
        normalized_frames.append({
            't': frame['t'],
            'landmarks': normalized_landmarks
        })
    
    return normalized_frames


def calculate_velocity_energy(landmarks_prev: np.ndarray, 
                               landmarks_curr: np.ndarray,
                               dt: float) -> float:
    """
    Calculate wrist velocity energy between two consecutive frames.
    
    Velocity energy = ||v||^2 where v = (p_curr - p_prev) / dt
    
    Args:
        landmarks_prev: Previous frame landmarks (21, 3)
        landmarks_curr: Current frame landmarks (21, 3)
        dt: Time delta between frames in seconds
        
    Returns:
        Squared L2 norm of wrist velocity
    """
    wrist_prev = landmarks_prev[0, :]
    wrist_curr = landmarks_curr[0, :]
    
    velocity = (wrist_curr - wrist_prev) / dt
    energy = np.dot(velocity, velocity)
    
    return float(energy)


def calculate_trajectory_velocity_energies(frames: List[Dict[str, Any]], 
                                            fps: float = 30.0) -> List[float]:
    """
    Calculate wrist velocity energy for each frame transition in a trajectory.
    
    Args:
        frames: List of frame dicts with 't' and 'landmarks' keys
        fps: Frames per second (default 30)
        
    Returns:
        List of velocity energies (one less than number of frames)
    """
    if len(frames) < 2:
        return []
    
    dt = 1.0 / fps
    energies = []
    
    for i in range(1, len(frames)):
        lm_prev = np.array(frames[i-1]['landmarks'], dtype=np.float64)
        lm_curr = np.array(frames[i]['landmarks'], dtype=np.float64)
        energy = calculate_velocity_energy(lm_prev, lm_curr, dt)
        energies.append(energy)
    
    return energies
