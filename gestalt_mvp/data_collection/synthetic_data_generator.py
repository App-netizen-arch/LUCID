#!/usr/bin/env python3
"""
Gestalt-MVP Synthetic Data Generator

Generates realistic synthetic gesture data for testing the pipeline
when real camera data is not available.

This creates trajectories that mimic the 10 gesture classes with
realistic variation in speed, amplitude, and noise.
"""

import json
import math
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np


# Gesture definitions - parametric trajectories for each class
# Each gesture is defined as a sequence of wrist positions and hand configurations
GESTURE_DEFINITIONS = {
    'G1': {  # Water: Chest → sharp arc to mouth → return to chest
        'name': 'Water',
        'duration_range': (0.8, 1.5),
        'wrist_path': [
            (0.0, 0.0, 0.5),   # Chest
            (0.0, 0.15, 0.4),  # Arc up
            (0.0, 0.2, 0.35),  # Mouth
            (0.0, 0.15, 0.4),  # Return arc
            (0.0, 0.0, 0.5),   # Back to chest
        ],
        'hand_shape': 'open',  # Open hand
    },
    'G2': {  # Help: Both hands at waist, palms up, lift 6in → return
        'name': 'Help',
        'duration_range': (1.0, 1.8),
        'wrist_path': [
            (-0.15, -0.2, 0.6),  # Left hand at waist
            (-0.15, -0.1, 0.6),  # Lift
            (-0.15, 0.0, 0.6),   # Higher
            (-0.15, -0.1, 0.6),  # Return
            (-0.15, -0.2, 0.6),  # Back
        ],
        'hand_shape': 'palms_up',
    },
    'G3': {  # Yes: Closed fist, sharp double-nod (up-down-up) → return
        'name': 'Yes',
        'duration_range': (0.6, 1.2),
        'wrist_path': [
            (0.0, 0.0, 0.5),
            (0.0, 0.05, 0.5),   # Up
            (0.0, -0.05, 0.5),  # Down
            (0.0, 0.05, 0.5),   # Up
            (0.0, 0.0, 0.5),    # Return
        ],
        'hand_shape': 'fist',
    },
    'G4': {  # No: Index+middle extended, horizontal snip → return
        'name': 'No',
        'duration_range': (0.5, 1.0),
        'wrist_path': [
            (0.0, 0.0, 0.5),
            (-0.1, 0.0, 0.5),   # Left
            (0.1, 0.0, 0.5),    # Right
            (0.0, 0.0, 0.5),    # Return
        ],
        'hand_shape': 'index_middle',
    },
    'G5': {  # Food: Bunched fingertips, tap chin twice → return to lap
        'name': 'Food',
        'duration_range': (1.0, 1.6),
        'wrist_path': [
            (0.0, -0.3, 0.7),   # Lap
            (0.0, 0.15, 0.4),   # Chin tap 1
            (0.0, 0.15, 0.4),   # Hold
            (0.0, 0.15, 0.4),   # Chin tap 2
            (0.0, -0.3, 0.7),   # Return to lap
        ],
        'hand_shape': 'bunched',
    },
    'G6': {  # Pain: Flat hand, palm in, tap chest center twice → return
        'name': 'Pain',
        'duration_range': (0.8, 1.4),
        'wrist_path': [
            (0.0, 0.0, 0.5),
            (0.0, 0.02, 0.45),  # Tap 1
            (0.0, 0.0, 0.5),
            (0.0, 0.02, 0.45),  # Tap 2
            (0.0, 0.0, 0.5),    # Return
        ],
        'hand_shape': 'flat_palm_in',
    },
    'G7': {  # Stop: Flat hand, palm out, sharp forward push → return
        'name': 'Stop',
        'duration_range': (0.6, 1.2),
        'wrist_path': [
            (0.0, 0.0, 0.5),
            (0.0, 0.0, 0.7),    # Push forward
            (0.0, 0.0, 0.5),    # Return
        ],
        'hand_shape': 'flat_palm_out',
    },
    'G8': {  # More: Both flat hands, palms facing, clap once → return
        'name': 'More',
        'duration_range': (0.7, 1.3),
        'wrist_path': [
            (-0.1, 0.0, 0.5),   # Left hand start
            (0.0, 0.0, 0.5),    # Clap
            (-0.1, 0.0, 0.5),   # Return
        ],
        'hand_shape': 'flat_palms_facing',
    },
    'G9': {  # Wait: Flat hand, palm down, slow horizontal circle → return
        'name': 'Wait',
        'duration_range': (1.2, 2.0),
        'wrist_path': [
            (0.0, 0.0, 0.5),
            (-0.05, 0.05, 0.5),  # Circle start
            (0.0, 0.1, 0.5),
            (0.05, 0.05, 0.5),
            (0.0, 0.0, 0.5),     # Return
        ],
        'hand_shape': 'flat_palm_down',
    },
    'G10': {  # Bathroom: Flat hand "C" shape, tap lower abdomen twice → return
        'name': 'Bathroom',
        'duration_range': (1.0, 1.6),
        'wrist_path': [
            (0.0, -0.25, 0.6),   # Start
            (0.0, -0.2, 0.55),   # Tap 1
            (0.0, -0.25, 0.6),
            (0.0, -0.2, 0.55),   # Tap 2
            (0.0, -0.25, 0.6),   # Return
        ],
        'hand_shape': 'c_shape',
    },
}

# Negative gesture definitions
NEGATIVE_DEFINITIONS = {
    'NEG-1': {  # Random: waving, shaking
        'name': 'Random',
        'duration_range': (0.5, 2.0),
        'pattern': 'random_walk',
    },
    'NEG-2': {  # Everyday: typing, drinking
        'name': 'Everyday',
        'duration_range': (0.8, 2.5),
        'pattern': 'everyday_motion',
    },
    'NEG-3': {  # Near-Miss: wiping mouth, scratching cheek
        'name': 'Near-Miss',
        'duration_range': (0.6, 1.5),
        'pattern': 'near_miss',
    },
}


def generate_hand_landmarks(base_pose: str, wrist_pos: tuple, 
                           frame_idx: int, total_frames: int,
                           noise_level: float = 0.02) -> List[List[float]]:
    """
    Generate 21 MediaPipe hand landmarks for a given hand shape and wrist position.
    
    MediaPipe landmark indices:
    0: Wrist
    1-4: Thumb
    5-8: Index finger
    9-12: Middle finger
    13-16: Ring finger
    17-20: Pinky
    
    Returns: List of 21 [x, y, z] coordinates
    """
    wx, wy, wz = wrist_pos
    
    # Base hand configuration templates (relative to wrist)
    hand_configs = {
        'open': [
            # Wrist
            [0.0, 0.0, 0.0],
            # Thumb (curved slightly)
            [0.02, -0.02, 0.02], [0.03, -0.03, 0.03], [0.04, -0.04, 0.04], [0.05, -0.05, 0.05],
            # Index (extended)
            [0.01, 0.02, 0.01], [0.01, 0.05, 0.01], [0.01, 0.08, 0.01], [0.01, 0.11, 0.01],
            # Middle (extended)
            [0.0, 0.02, 0.0], [0.0, 0.05, 0.0], [0.0, 0.09, 0.0], [0.0, 0.12, 0.0],
            # Ring (extended)
            [-0.01, 0.02, 0.01], [-0.01, 0.05, 0.01], [-0.01, 0.08, 0.01], [-0.01, 0.11, 0.01],
            # Pinky (extended)
            [-0.02, 0.02, 0.02], [-0.02, 0.04, 0.02], [-0.02, 0.06, 0.02], [-0.02, 0.08, 0.02],
        ],
        'fist': [
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.02], [0.03, 0.0, 0.03], [0.03, 0.0, 0.02], [0.02, 0.0, 0.01],
            [0.01, 0.0, 0.01], [0.01, 0.02, 0.01], [0.01, 0.03, 0.01], [0.01, 0.03, 0.01],
            [0.0, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.03, 0.0], [0.0, 0.03, 0.0],
            [-0.01, 0.0, 0.01], [-0.01, 0.02, 0.01], [-0.01, 0.03, 0.01], [-0.01, 0.03, 0.01],
            [-0.02, 0.0, 0.02], [-0.02, 0.02, 0.02], [-0.02, 0.03, 0.02], [-0.02, 0.03, 0.02],
        ],
        'palms_up': [
            [0.0, 0.0, 0.0],
            [0.02, -0.01, -0.02], [0.03, -0.02, -0.03], [0.04, -0.02, -0.03], [0.05, -0.02, -0.03],
            [0.01, 0.01, -0.02], [0.01, 0.04, -0.02], [0.01, 0.07, -0.02], [0.01, 0.10, -0.02],
            [0.0, 0.01, -0.02], [0.0, 0.04, -0.02], [0.0, 0.08, -0.02], [0.0, 0.11, -0.02],
            [-0.01, 0.01, -0.02], [-0.01, 0.04, -0.02], [-0.01, 0.07, -0.02], [-0.01, 0.10, -0.02],
            [-0.02, 0.01, -0.02], [-0.02, 0.03, -0.02], [-0.02, 0.05, -0.02], [-0.02, 0.07, -0.02],
        ],
        'index_middle': [
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.02], [0.03, 0.0, 0.03], [0.03, 0.0, 0.02], [0.02, 0.0, 0.01],
            [0.01, 0.01, 0.01], [0.01, 0.04, 0.01], [0.01, 0.08, 0.01], [0.01, 0.11, 0.01],
            [0.0, 0.01, 0.0], [0.0, 0.04, 0.0], [0.0, 0.08, 0.0], [0.0, 0.11, 0.0],
            [-0.01, 0.01, 0.02], [-0.01, 0.02, 0.02], [-0.01, 0.03, 0.02], [-0.01, 0.03, 0.02],
            [-0.02, 0.01, 0.03], [-0.02, 0.02, 0.03], [-0.02, 0.03, 0.03], [-0.02, 0.03, 0.03],
        ],
        'bunched': [
            [0.0, 0.0, 0.0],
            [0.02, 0.01, 0.02], [0.02, 0.02, 0.03], [0.02, 0.02, 0.03], [0.02, 0.02, 0.03],
            [0.01, 0.02, 0.02], [0.01, 0.04, 0.02], [0.01, 0.05, 0.02], [0.00, 0.05, 0.02],
            [0.0, 0.02, 0.01], [0.0, 0.04, 0.01], [0.0, 0.05, 0.01], [-0.01, 0.05, 0.01],
            [-0.01, 0.02, 0.02], [-0.01, 0.04, 0.02], [-0.01, 0.05, 0.02], [-0.02, 0.05, 0.02],
            [-0.02, 0.02, 0.03], [-0.02, 0.03, 0.03], [-0.02, 0.04, 0.03], [-0.03, 0.04, 0.03],
        ],
        'flat_palm_in': [
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.03], [0.03, 0.0, 0.04], [0.04, 0.0, 0.04], [0.05, 0.0, 0.04],
            [0.01, 0.01, 0.03], [0.01, 0.04, 0.03], [0.01, 0.07, 0.03], [0.01, 0.10, 0.03],
            [0.0, 0.01, 0.03], [0.0, 0.04, 0.03], [0.0, 0.08, 0.03], [0.0, 0.11, 0.03],
            [-0.01, 0.01, 0.03], [-0.01, 0.04, 0.03], [-0.01, 0.07, 0.03], [-0.01, 0.10, 0.03],
            [-0.02, 0.01, 0.03], [-0.02, 0.03, 0.03], [-0.02, 0.05, 0.03], [-0.02, 0.07, 0.03],
        ],
        'flat_palm_out': [
            [0.0, 0.0, 0.0],
            [0.02, 0.0, -0.03], [0.03, 0.0, -0.04], [0.04, 0.0, -0.04], [0.05, 0.0, -0.04],
            [0.01, 0.01, -0.03], [0.01, 0.04, -0.03], [0.01, 0.07, -0.03], [0.01, 0.10, -0.03],
            [0.0, 0.01, -0.03], [0.0, 0.04, -0.03], [0.0, 0.08, -0.03], [0.0, 0.11, -0.03],
            [-0.01, 0.01, -0.03], [-0.01, 0.04, -0.03], [-0.01, 0.07, -0.03], [-0.01, 0.10, -0.03],
            [-0.02, 0.01, -0.03], [-0.02, 0.03, -0.03], [-0.02, 0.05, -0.03], [-0.02, 0.07, -0.03],
        ],
        'flat_palms_facing': [
            [0.0, 0.0, 0.0],
            [0.03, 0.0, 0.0], [0.05, 0.0, 0.0], [0.06, 0.0, 0.0], [0.07, 0.0, 0.0],
            [0.01, 0.01, 0.0], [0.01, 0.04, 0.0], [0.01, 0.07, 0.0], [0.01, 0.10, 0.0],
            [0.0, 0.01, 0.0], [0.0, 0.04, 0.0], [0.0, 0.08, 0.0], [0.0, 0.11, 0.0],
            [-0.01, 0.01, 0.0], [-0.01, 0.04, 0.0], [-0.01, 0.07, 0.0], [-0.01, 0.10, 0.0],
            [-0.02, 0.01, 0.0], [-0.02, 0.03, 0.0], [-0.02, 0.05, 0.0], [-0.02, 0.07, 0.0],
        ],
        'flat_palm_down': [
            [0.0, 0.0, 0.0],
            [0.02, 0.0, 0.02], [0.03, 0.0, 0.03], [0.04, 0.0, 0.03], [0.05, 0.0, 0.03],
            [0.01, 0.01, 0.02], [0.01, 0.04, 0.02], [0.01, 0.07, 0.02], [0.01, 0.10, 0.02],
            [0.0, 0.01, 0.02], [0.0, 0.04, 0.02], [0.0, 0.08, 0.02], [0.0, 0.11, 0.02],
            [-0.01, 0.01, 0.02], [-0.01, 0.04, 0.02], [-0.01, 0.07, 0.02], [-0.01, 0.10, 0.02],
            [-0.02, 0.01, 0.02], [-0.02, 0.03, 0.02], [-0.02, 0.05, 0.02], [-0.02, 0.07, 0.02],
        ],
        'c_shape': [
            [0.0, 0.0, 0.0],
            [0.03, 0.0, 0.02], [0.04, 0.01, 0.03], [0.04, 0.02, 0.03], [0.03, 0.03, 0.03],
            [0.01, 0.02, 0.02], [0.01, 0.05, 0.02], [0.01, 0.07, 0.02], [0.00, 0.08, 0.02],
            [0.0, 0.02, 0.02], [0.0, 0.05, 0.02], [0.0, 0.08, 0.02], [0.0, 0.10, 0.02],
            [-0.01, 0.02, 0.02], [-0.01, 0.05, 0.02], [-0.01, 0.07, 0.02], [-0.02, 0.08, 0.02],
            [-0.02, 0.02, 0.02], [-0.02, 0.04, 0.02], [-0.02, 0.05, 0.02], [-0.03, 0.06, 0.02],
        ],
    }
    
    # Get base configuration
    if base_pose not in hand_configs:
        base_pose = 'open'
    
    landmarks = hand_configs[base_pose].copy()
    
    # Add temporal variation for dynamic gestures
    progress = frame_idx / max(total_frames - 1, 1)
    
    # Subtle finger motion during gesture
    for i in range(1, 21):
        motion_amplitude = 0.005 * math.sin(progress * math.pi * 3)
        landmarks[i][1] += motion_amplitude
    
    # Add Gaussian noise
    for i in range(21):
        for j in range(3):
            landmarks[i][j] += np.random.normal(0, noise_level)
    
    # Translate to wrist position
    for i in range(21):
        landmarks[i][0] += wx
        landmarks[i][1] += wy
        landmarks[i][2] += wz
    
    return landmarks


def interpolate_path(path: List[tuple], num_points: int) -> List[tuple]:
    """Interpolate a path to have exactly num_points."""
    if len(path) < 2:
        return path
    
    # Calculate cumulative distances
    distances = [0.0]
    for i in range(1, len(path)):
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        dz = path[i][2] - path[i-1][2]
        distances.append(distances[-1] + math.sqrt(dx*dx + dy*dy + dz*dz))
    
    total_dist = distances[-1]
    
    # Interpolate
    interpolated = []
    for i in range(num_points):
        target_dist = (i / max(num_points - 1, 1)) * total_dist
        
        # Find segment
        seg_idx = 0
        for j in range(len(distances) - 1):
            if distances[j] <= target_dist < distances[j + 1]:
                seg_idx = j
                break
        
        # Interpolate within segment
        if seg_idx < len(path) - 1:
            seg_start_dist = distances[seg_idx]
            seg_end_dist = distances[seg_idx + 1]
            seg_frac = (target_dist - seg_start_dist) / max(seg_end_dist - seg_start_dist, 0.001)
            
            x = path[seg_idx][0] + seg_frac * (path[seg_idx + 1][0] - path[seg_idx][0])
            y = path[seg_idx][1] + seg_frac * (path[seg_idx + 1][1] - path[seg_idx][1])
            z = path[seg_idx][2] + seg_frac * (path[seg_idx + 1][2] - path[seg_idx][2])
            interpolated.append((x, y, z))
        else:
            interpolated.append(path[-1])
    
    return interpolated


def generate_negative_trajectory(pattern: str, duration_s: float, fps: int = 30) -> List[List[List[float]]]:
    """Generate a negative (non-gesture) trajectory."""
    num_frames = int(duration_s * fps)
    frames = []
    
    if pattern == 'random_walk':
        # Random waving motion
        pos = [0.0, 0.0, 0.5]
        for i in range(num_frames):
            pos[0] += np.random.normal(0, 0.02)  # X jitter
            pos[1] += np.random.normal(0, 0.03)  # Y jitter
            pos[2] += np.random.normal(0, 0.01)  # Z jitter
            
            # Clamp to reasonable bounds
            pos[0] = max(-0.3, min(0.3, pos[0]))
            pos[1] = max(-0.3, min(0.3, pos[1]))
            pos[2] = max(0.3, min(0.8, pos[2]))
            
            landmarks = generate_hand_landmarks('open', tuple(pos), i, num_frames)
            frames.append(landmarks)
    
    elif pattern == 'everyday_motion':
        # Typing-like motion
        base_x = np.random.uniform(-0.15, 0.15)
        base_y = np.random.uniform(-0.2, -0.1)
        base_z = np.random.uniform(0.6, 0.7)
        
        for i in range(num_frames):
            # Alternating small motions
            offset_x = 0.03 * math.sin(i * 0.5)
            offset_y = 0.02 * math.cos(i * 0.3)
            
            pos = (base_x + offset_x, base_y + offset_y, base_z)
            landmarks = generate_hand_landmarks('open', pos, i, num_frames)
            frames.append(landmarks)
    
    elif pattern == 'near_miss':
        # Face-touching motions (similar to real gestures but not quite)
        t = np.linspace(0, 1, num_frames)
        
        # Scratching cheek or wiping mouth
        start = (np.random.uniform(-0.1, 0.1), np.random.uniform(0.1, 0.2), 0.4)
        end = (start[0] + np.random.uniform(-0.05, 0.05), 
               start[1] + np.random.uniform(-0.05, 0.05), 
               start[2])
        
        for i in range(num_frames):
            progress = t[i]
            # Non-linear motion (hesitation in middle)
            progress = 0.5 - 0.5 * math.cos(progress * math.pi)
            
            x = start[0] + progress * (end[0] - start[0])
            y = start[1] + progress * (end[1] - start[1])
            z = start[2] + progress * (end[2] - start[2])
            
            landmarks = generate_hand_landmarks('open', (x, y, z), i, num_frames)
            frames.append(landmarks)
    
    return frames


def generate_gesture_trajectory(gesture_id: str, condition: str = 'normal',
                                fps: int = 30) -> Tuple[List[List[List[float]]], float]:
    """
    Generate a synthetic gesture trajectory.
    
    Args:
        gesture_id: G1-G10 or NEG-1/NEG-2/NEG-3
        condition: normal, fast-small, slow-large, 45-degree, occlusion
        fps: Frames per second (default 30)
    
    Returns:
        Tuple of (frames, duration_s)
    """
    # Determine gesture definition
    if gesture_id.startswith('NEG'):
        defn = NEGATIVE_DEFINITIONS.get(gesture_id, NEGATIVE_DEFINITIONS['NEG-1'])
        duration_s = np.random.uniform(*defn['duration_range'])
        frames = generate_negative_trajectory(defn['pattern'], duration_s, fps)
        return frames, duration_s
    
    defn = GESTURE_DEFINITIONS.get(gesture_id)
    if not defn:
        raise ValueError(f"Unknown gesture: {gesture_id}")
    
    # Base duration with condition modification
    base_duration = np.random.uniform(*defn['duration_range'])
    
    condition_modifiers = {
        'normal': 1.0,
        'fast-small': 0.7,      # Faster, smaller amplitude
        'slow-large': 1.5,      # Slower, larger amplitude
        '45-degree': 1.0,       # Same speed, rotated
        'occlusion': 1.0,       # Same speed, with missing frames
    }
    
    duration_s = base_duration * condition_modifiers.get(condition, 1.0)
    num_frames = int(duration_s * fps)
    
    # Get base wrist path
    base_path = defn['wrist_path']
    
    # Apply condition modifications
    if condition == 'fast-small':
        # Reduce amplitude
        base_path = [(x * 0.7, y * 0.7, z) for x, y, z in base_path]
    elif condition == 'slow-large':
        # Increase amplitude
        base_path = [(x * 1.3, y * 1.3, z) for x, y, z in base_path]
    elif condition == '45-degree':
        # Rotate around Z axis by 45 degrees
        cos_45 = math.cos(math.pi / 4)
        sin_45 = math.sin(math.pi / 4)
        base_path = [
            (x * cos_45 - y * sin_45, x * sin_45 + y * cos_45, z)
            for x, y, z in base_path
        ]
    
    # Interpolate path to match number of frames
    wrist_positions = interpolate_path(base_path, num_frames)
    
    # Generate full landmark trajectories
    frames = []
    hand_shape = defn.get('hand_shape', 'open')
    
    for i, wrist_pos in enumerate(wrist_positions):
        landmarks = generate_hand_landmarks(hand_shape, wrist_pos, i, num_frames)
        
        # Apply occlusion condition (simulate missing landmarks)
        if condition == 'occlusion' and i > num_frames // 3 and i < 2 * num_frames // 3:
            # Occlude some fingers (set to zero or NaN-like values)
            # In practice, MediaPipe would still estimate, so add large noise
            for j in range(17, 21):  # Pinky occluded
                for k in range(3):
                    landmarks[j][k] += np.random.normal(0, 0.1)
        
        frames.append(landmarks)
    
    return frames, duration_s


def generate_synthetic_segment(session: int, gesture_id: str, repetition: int,
                               condition: str = 'normal') -> Dict[str, Any]:
    """Generate a complete synthetic gesture segment."""
    frames, duration_s = generate_gesture_trajectory(gesture_id, condition)
    
    # Build JSON structure matching the spec
    trajectory_data = {
        'session': session,
        'gesture_id': gesture_id,
        'repetition': repetition,
        'condition': condition,
        'frames': [
            {
                't': round(i / 30.0, 3),
                'landmarks': [[round(c, 6) for c in lm] for lm in frame]
            }
            for i, frame in enumerate(frames)
        ],
        'duration_s': round(duration_s, 3),
        'frame_count': len(frames)
    }
    
    return trajectory_data


def generate_full_dataset(output_dir: Path, 
                          reps_per_gesture: int = 8,
                          include_negatives: bool = True) -> None:
    """
    Generate a complete synthetic dataset for all 5 sessions.
    
    Sessions 1-2: Enrollment (10 gestures × 8 reps × 5 conditions = 400 segments)
    Session 3: Calibration (same)
    Sessions 4-5: Locked test (same)
    Plus negative samples in each session.
    """
    gestures = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10']
    conditions = ['normal', 'fast-small', 'slow-large', '45-degree', 'occlusion']
    negatives = ['NEG-1', 'NEG-2', 'NEG-3'] if include_negatives else []
    
    logger.info(f"Generating synthetic dataset in {output_dir}")
    
    for session in range(1, 6):
        session_dir = output_dir / f"session{session}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"  Session {session}: Generating gestures...")
        
        # Generate positive gestures
        for gesture_id in gestures:
            for rep in range(1, reps_per_gesture + 1):
                # Vary conditions across repetitions
                condition = conditions[(rep - 1) % len(conditions)]
                
                segment = generate_synthetic_segment(
                    session=session,
                    gesture_id=gesture_id,
                    repetition=rep,
                    condition=condition
                )
                
                filename = f"session{session}_gesture{gesture_id}_rep{rep}.json"
                filepath = session_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump(segment, f, indent=2)
        
        # Generate negative samples (20 per negative class per session)
        if include_negatives:
            logger.info(f"  Session {session}: Generating negative samples...")
            for neg_class in negatives:
                for rep in range(1, 21):
                    segment = generate_synthetic_segment(
                        session=session,
                        gesture_id=neg_class,
                        repetition=rep,
                        condition='normal'
                    )
                    
                    filename = f"session{session}_{neg_class}_rep{rep}.json"
                    filepath = session_dir / filename
                    
                    with open(filepath, 'w') as f:
                        json.dump(segment, f, indent=2)
        
        # Count files
        file_count = len(list(session_dir.glob("*.json")))
        logger.info(f"  Session {session}: Generated {file_count} segments")
    
    logger.info(f"Dataset generation complete!")


if __name__ == '__main__':
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Generate dataset
    data_dir = Path(__file__).parent / 'data'
    generate_full_dataset(data_dir, reps_per_gesture=8, include_negatives=True)
    
    print(f"\nSynthetic dataset generated successfully!")
    print(f"Location: {data_dir}")
    print(f"\nTo evaluate, run:")
    print(f"  cd ../evaluation && python evaluate.py --data-dir {data_dir}")
