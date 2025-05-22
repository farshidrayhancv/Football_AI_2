"""
Player possession detection module
"""

import cv2
import numpy as np
import supervision as sv
from supervision.geometry.core import Position


class PossessionDetector:
    """Detect which player has possession of the ball."""
    
    def __init__(self, proximity_threshold=50, possession_frames=3, highlight_color=None):
        """
        Initialize possession detector.
        
        Args:
            proximity_threshold: Distance threshold for possession detection
            possession_frames: Number of frames required for stable possession
            highlight_color: Color to highlight player with possession
        """
        self.proximity_threshold = proximity_threshold
        self.possession_frames = possession_frames
        
        # Set highlight color
        if highlight_color is None:
            self.highlight_color = sv.Color(0, 255, 0)
        else:
            self.highlight_color = highlight_color
        
        # Track possession state
        self.current_player_id = None
        self.current_team_id = None
        self.candidate_player_id = None
        self.candidate_team_id = None
        self.candidate_counter = 0
        self.ball_position_history = []
    
    def update(self, detections, ball_position, transformer=None):
        """
        Update possession detection based on new frame data.
        
        Args:
            detections: Detections object with player information
            ball_position: Position of the ball [x, y]
            transformer: Optional transformer to convert positions to pitch coordinates
            
        Returns:
            Possession information dictionary
        """
        if len(detections) == 0 or ball_position is None:
            return {"player_id": self.current_player_id, "team_id": self.current_team_id}
        
        # Transform positions if transformer provided
        if transformer is not None:
            # Transform ball position
            ball_position_transformed = transformer.transform_points(np.array([ball_position]))[0]
            
            # Get player positions
            player_positions = detections.get_anchors_coordinates(Position.BOTTOM_CENTER)
            
            # Transform player positions
            player_positions_transformed = transformer.transform_points(player_positions)
            
            # Use transformed positions for distance calculation
            distances = np.linalg.norm(player_positions_transformed - ball_position_transformed, axis=1)
        else:
            # Use original positions for distance calculation
            player_positions = detections.get_anchors_coordinates(Position.BOTTOM_CENTER)
            distances = np.linalg.norm(player_positions - ball_position, axis=1)
        
        # Find the closest player
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        
        # Check if the closest player is within threshold
        if closest_distance > self.proximity_threshold:
            # No player has possession
            self.candidate_counter = 0
            self.candidate_player_id = None
            self.candidate_team_id = None
            
            return {"player_id": self.current_player_id, "team_id": self.current_team_id}
        
        # Get player information
        closest_player_id = int(detections.tracker_id[closest_idx])
        closest_team_id = int(detections.class_id[closest_idx])
        
        # Check if this is the same candidate
        if closest_player_id == self.candidate_player_id:
            self.candidate_counter += 1
        else:
            # New candidate
            self.candidate_player_id = closest_player_id
            self.candidate_team_id = closest_team_id
            self.candidate_counter = 1
        
        # Check if candidate has been consistent for enough frames
        if self.candidate_counter >= self.possession_frames:
            self.current_player_id = self.candidate_player_id
            self.current_team_id = self.candidate_team_id
        
        # Track ball position
        self.ball_position_history.append(ball_position)
        if len(self.ball_position_history) > 30:  # Keep last 30 frames
            self.ball_position_history.pop(0)
        
        # Return possession information
        return {
            "player_id": self.current_player_id,
            "team_id": self.current_team_id,
            "distance": closest_distance,
            "candidate_player_id": self.candidate_player_id,
            "candidate_counter": self.candidate_counter
        }
    
    def highlight_possession(self, frame, detections, possession_info=None):
        """
        Highlight the player with possession on the frame.
        
        Args:
            frame: Video frame to annotate
            detections: Detections object with player information
            possession_info: Optional possession information
            
        Returns:
            Annotated frame
        """
        # Use current possession if not provided
        if possession_info is None:
            player_id = self.current_player_id
            team_id = self.current_team_id
        else:
            player_id = possession_info.get("player_id")
            team_id = possession_info.get("team_id")
        
        if player_id is None or len(detections) == 0:
            return frame
        
        # Find the player with possession
        player_idx = None
        for i, track_id in enumerate(detections.tracker_id):
            if int(track_id) == player_id:
                player_idx = i
                break
        
        if player_idx is None:
            return frame
        
        # Extract bounding box
        bbox = detections.xyxy[player_idx].astype(int)
        
        # Create a copy of the frame
        annotated = frame.copy()
        
        # Draw thicker box around player with possession
        cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                     self.highlight_color.as_bgr(), 3)
        
        # Add possession indicator
        cv2.putText(annotated, "POSSESSION", (bbox[0], bbox[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.highlight_color.as_bgr(), 2)
        
        # Add circle above player
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = bbox[1] - 30
        cv2.circle(annotated, (center_x, center_y), 15, self.highlight_color.as_bgr(), -1)
        
        # Add player ID
        cv2.putText(annotated, f"{player_id}", (center_x - 8, center_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated
    
    def get_possession_stats(self):
        """Get possession statistics."""
        return {
            "current_player_id": self.current_player_id,
            "current_team_id": self.current_team_id
        }
