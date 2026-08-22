#!/usr/bin/env python3
"""
Gestalt-MVP Data Collection Script

Captures hand gesture data using MediaPipe Hands, applies normalization,
and saves isolated trajectory segments as JSON files.

Usage:
    python collect.py --session 1 --gesture G1 --repetition 1 --condition normal

Conditions:
    - normal: Standard execution
    - fast-small: Faster, smaller amplitude
    - slow-large: Slower, larger amplitude  
    - 45-degree: Camera rotated 45 degrees
    - occlusion: Partial hand occlusion

File naming: session{N}_gesture{G#}_rep{R}.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import mediapipe as mp
import numpy as np

from normalize import normalize_trajectory, calculate_trajectory_velocity_energies
from segment import (
    segment_trajectory, 
    validate_segment_quality,
    DEFAULT_TAU_START,
    DEFAULT_TAU_REST,
    MIN_DURATION_S,
    MAX_DURATION_S,
    FRAMES_FOR_START,
    FRAMES_FOR_REST
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_collection.log')
    ]
)
logger = logging.getLogger(__name__)


# Gesture vocabulary (frozen from Section 7)
VALID_GESTURES = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10']
VALID_CONDITIONS = ['normal', 'fast-small', 'slow-large', '45-degree', 'occlusion']

# MediaPipe configuration
MEDIAPIPE_MODEL_COMPLEXITY = 1
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5
FPS = 30.0
FRAME_TIME = 1.0 / FPS


class HandGestureCollector:
    """Collects and processes hand gesture data using MediaPipe."""
    
    def __init__(self, session_id: int, gesture_id: str, repetition: int,
                 condition: str, output_dir: str):
        """
        Initialize the collector.
        
        Args:
            session_id: Session number (1-5)
            gesture_id: Gesture identifier (G1-G10)
            repetition: Repetition number
            condition: Recording condition
            output_dir: Directory to save JSON files
        """
        self.session_id = session_id
        self.gesture_id = gesture_id
        self.repetition = repetition
        self.condition = condition
        self.output_dir = Path(output_dir)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
            max_num_hands=2  # Support two-handed gestures
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Frame buffer for continuous recording
        self.frames_buffer: List[Dict[str, Any]] = []
        self.is_recording = False
        self.start_time = 0.0
        
        # State tracking
        self.last_landmarks: Optional[np.ndarray] = None
        self.segment_found = False
        
        logger.info(f"Initialized collector: Session {session_id}, "
                   f"Gesture {gesture_id}, Repetition {repetition}, "
                   f"Condition {condition}")
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[List[List[float]]]:
        """
        Extract hand landmarks from an image frame.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            List of 21 [x, y, z] landmark coordinates, or None if no hand detected
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_image)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Use the first detected hand (or could implement hand selection logic)
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract 21 landmarks as [x, y, z] coordinates
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        
        return landmarks
    
    def process_frame(self, image: np.ndarray, timestamp: float) -> bool:
        """
        Process a single video frame.
        
        Args:
            image: BGR image frame
            timestamp: Frame timestamp in seconds
            
        Returns:
            True if hand detected, False otherwise
        """
        landmarks = self.extract_landmarks(image)
        
        if landmarks is None:
            # No hand detected
            if self.is_recording:
                # Add empty frame to maintain timing
                self.frames_buffer.append({
                    't': timestamp,
                    'landmarks': None
                })
            return False
        
        # Normalize landmarks
        normalized_landmarks = normalize_trajectory([{
            't': timestamp,
            'landmarks': landmarks
        }])[0]['landmarks']
        
        frame_data = {
            't': timestamp,
            'landmarks': normalized_landmarks,
            'raw_landmarks': landmarks  # Keep raw for debugging
        }
        
        if self.is_recording:
            self.frames_buffer.append(frame_data)
        
        return True
    
    def check_segmentation(self) -> Optional[List[Dict[str, Any]]]:
        """
        Check if a valid gesture segment has been captured.
        
        Returns:
            Segmented frames if valid gesture found, None otherwise
        """
        if len(self.frames_buffer) < 20:  # Minimum frames for segmentation
            return None
        
        # Filter out frames without landmarks
        valid_frames = [f for f in self.frames_buffer if f['landmarks'] is not None]
        
        if len(valid_frames) < 20:
            return None
        
        segment, error = segment_trajectory(
            valid_frames,
            tau_start=DEFAULT_TAU_START,
            tau_rest=DEFAULT_TAU_REST,
            fps=FPS,
            min_duration=MIN_DURATION_S,
            max_duration=MAX_DURATION_S
        )
        
        if segment is not None:
            quality = validate_segment_quality(segment, FPS)
            logger.info(f"Segment found: duration={quality['duration_s']:.2f}s, "
                       f"frames={quality['frame_count']}, "
                       f"return_to_rest={quality['has_return_to_rest']}")
            return segment
        
        return None
    
    def save_segment(self, segment: List[Dict[str, Any]]) -> str:
        """
        Save a segmented gesture to JSON file.
        
        Args:
            segment: List of frame dicts
            
        Returns:
            Path to saved file
        """
        # Prepare output data according to schema
        duration = segment[-1]['t'] - segment[0]['t']
        
        output_data = {
            'session': self.session_id,
            'gesture_id': self.gesture_id,
            'repetition': self.repetition,
            'condition': self.condition,
            'frames': segment,
            'duration_s': round(duration, 3),
            'frame_count': len(segment)
        }
        
        # Generate filename
        filename = f"session{self.session_id}_gesture{self.gesture_id}_rep{self.repetition}.json"
        filepath = self.output_dir / filename
        
        # Write JSON file
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved segment to {filepath}")
        return str(filepath)
    
    def collect_gesture(self, timeout: float = 30.0) -> Optional[str]:
        """
        Collect a single gesture from the camera.
        
        Opens camera, displays preview, waits for user to perform gesture,
        automatically segments and saves.
        
        Args:
            timeout: Maximum time to wait for gesture in seconds
            
        Returns:
            Path to saved file, or None if failed/timed out
        """
        # Open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return None
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        
        logger.info(f"Camera opened. Press 'q' to quit, 's' to start manual recording.")
        logger.info(f"Perform gesture '{self.gesture_id}' ({self.condition})")
        
        start_time = time.time()
        frame_timestamp = 0.0
        auto_recording_started = False
        manual_mode = False
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    logger.warning("Failed to read frame")
                    continue
                
                current_time = time.time()
                elapsed = current_time - start_time
                
                if elapsed > timeout:
                    logger.warning(f"Timeout after {timeout}s")
                    break
                
                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("User quit")
                    break
                
                if key == ord('s'):
                    manual_mode = True
                    if not self.is_recording:
                        logger.info("Manual recording started")
                        self.is_recording = True
                        self.frames_buffer = []
                        self.start_time = current_time
                        frame_timestamp = 0.0
                
                if key == ord(' '):  # Space bar to end manual recording
                    if manual_mode and self.is_recording:
                        logger.info("Manual recording ended")
                        self.is_recording = False
                        
                        # Try to segment
                        segment = self.check_segmentation()
                        if segment:
                            filepath = self.save_segment(segment)
                            self.segment_found = True
                            break
                        else:
                            logger.warning("No valid segment found, retry")
                            self.frames_buffer = []
                
                # Process frame
                hand_detected = self.process_frame(frame, frame_timestamp)
                frame_timestamp += FRAME_TIME
                
                # Auto-start recording when hand detected and motion begins
                if not manual_mode and hand_detected and not self.is_recording:
                    # Calculate velocity energy to detect motion start
                    if len(self.frames_buffer) >= 2:
                        energies = calculate_trajectory_velocity_energies(
                            self.frames_buffer[-5:], FPS
                        )
                        if energies and np.mean(energies[-3:]) > DEFAULT_TAU_START:
                            logger.info("Auto-starting recording (motion detected)")
                            self.is_recording = True
                            self.frames_buffer = []  # Clear pre-motion frames
                            self.start_time = current_time
                            frame_timestamp = 0.0
                            auto_recording_started = True
                
                # Check for valid segment during recording
                if self.is_recording and auto_recording_started:
                    segment = self.check_segmentation()
                    if segment:
                        filepath = self.save_segment(segment)
                        self.segment_found = True
                        break
                    
                    # Check if recording too long
                    if frame_timestamp > MAX_DURATION_S + 1.0:
                        logger.warning("Recording too long, resetting")
                        self.is_recording = False
                        self.frames_buffer = []
                        auto_recording_started = False
                
                # Display frame with status
                display_frame = frame.copy()
                
                # Add status text
                status = "WAITING"
                if self.is_recording:
                    status = f"RECORDING ({frame_timestamp:.1f}s)"
                if self.segment_found:
                    status = "SEGMENT FOUND"
                
                cv2.putText(display_frame, f"Gesture: {self.gesture_id}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Condition: {self.condition}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Status: {status}",
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (255, 0, 0) if self.is_recording else (0, 255, 0), 2)
                cv2.putText(display_frame, f"Elapsed: {elapsed:.1f}s / {timeout}s",
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Draw hand landmarks if detected
                if hand_detected:
                    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(rgb_frame)
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(
                                display_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                            )
                
                cv2.imshow('Gestalt Data Collection', display_frame)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        if self.segment_found:
            logger.info("Gesture collection successful")
            return str(self.output_dir / f"session{self.session_id}_gesture{self.gesture_id}_rep{self.repetition}.json")
        else:
            logger.warning("Gesture collection incomplete")
            return None
    
    def cleanup(self):
        """Clean up resources."""
        self.hands.close()
        logger.info("Collector cleaned up")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Gestalt-MVP Data Collection Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python collect.py --session 1 --gesture G1 --repetition 1 --condition normal
    python collect.py --session 2 --gesture G5 --repetition 3 --condition fast-small
        """
    )
    
    parser.add_argument('--session', type=int, required=True,
                       help='Session number (1-5)')
    parser.add_argument('--gesture', type=str, required=True,
                       choices=VALID_GESTURES,
                       help='Gesture ID (G1-G10)')
    parser.add_argument('--repetition', type=int, required=True,
                       help='Repetition number (1-10)')
    parser.add_argument('--condition', type=str, default='normal',
                       choices=VALID_CONDITIONS,
                       help='Recording condition')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for JSON files')
    parser.add_argument('--timeout', type=float, default=30.0,
                       help='Maximum time to wait for gesture (seconds)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate inputs
    if args.session < 1 or args.session > 5:
        logger.error("Session must be between 1 and 5")
        sys.exit(1)
    
    if args.repetition < 1 or args.repetition > 10:
        logger.error("Repetition must be between 1 and 10")
        sys.exit(1)
    
    # Create collector
    collector = HandGestureCollector(
        session_id=args.session,
        gesture_id=args.gesture,
        repetition=args.repetition,
        condition=args.condition,
        output_dir=args.output_dir
    )
    
    try:
        # Collect gesture
        result = collector.collect_gesture(timeout=args.timeout)
        
        if result:
            logger.info(f"Successfully saved: {result}")
            sys.exit(0)
        else:
            logger.warning("Collection failed or cancelled")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    
    finally:
        collector.cleanup()


if __name__ == '__main__':
    main()
