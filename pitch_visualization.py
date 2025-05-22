"""
Pitch visualization module for tactical view
"""

import cv2
import numpy as np
import supervision as sv
from supervision.geometry.core import Position
from sports.annotators.soccer import (
    draw_pitch,
    draw_points_on_pitch,
    draw_paths_on_pitch
)


class PitchVisualizer:
    """Visualize players and ball on a tactical pitch view."""
    
    def __init__(self, config, pitch_config):
        """
        Initialize pitch visualizer.
        
        Args:
            config: Configuration dictionary
            pitch_config: Soccer pitch configuration
        """
        self.config = config
        self.pitch_config = pitch_config
        
        # Initialize colors
        self.team_colors = {
            0: sv.Color.from_hex(config['visualization']['team_colors']['team_1']),
            1: sv.Color.from_hex(config['visualization']['team_colors']['team_2']),
            2: sv.Color.from_hex(config['visualization']['team_colors']['referee'])
        }
        self.ball_color = sv.Color.from_hex(config['visualization']['ball_color'])
        self.possession_color = sv.Color.from_hex(config['possession']['highlight_color'])
        
        # Initialize dimensions
        self.width = config['visualization']['pitch_view_width']
        self.height = config['visualization']['pitch_view_height']
        
        # Ball trail
        self.ball_trail = []
        self.max_trail_length = 30
    
    def render(self, person_detections, ball_detections, transformer, possession_info=None):
        """
        Render tactical pitch view.
        
        Args:
            person_detections: Detections object with player information
            ball_detections: Detections object with ball information
            transformer: Transformer for coordinate conversion
            possession_info: Optional possession information
            
        Returns:
            Rendered pitch view
        """
        # Create base pitch image
        pitch_view = draw_pitch(
            config=self.pitch_config,
            width=self.width,
            height=self.height
        )
        
        # Draw players if available
        if len(person_detections) > 0:
            # Get player positions in frame coordinates
            player_positions = person_detections.get_anchors_coordinates(Position.BOTTOM_CENTER)
            
            # Transform to pitch coordinates
            pitch_positions = transformer.transform_points(player_positions)
            
            # Draw players by team
            for team_id in set(person_detections.class_id):
                team_mask = person_detections.class_id == team_id
                team_positions = pitch_positions[team_mask]
                
                if len(team_positions) > 0:
                    # Get team color
                    color = self.team_colors.get(int(team_id), sv.Color.WHITE)
                    
                    # Draw team players
                    pitch_view = draw_points_on_pitch(
                        config=self.pitch_config,
                        xy=team_positions,
                        face_color=color,
                        edge_color=sv.Color.BLACK,
                        radius=12,
                        pitch=pitch_view
                    )
                    
                    # Add player IDs
                    team_ids = person_detections.tracker_id[team_mask]
                    for pos, player_id in zip(team_positions, team_ids):
                        x, y = int(pos[0]), int(pos[1])
                        # Transform to image coordinates
                        img_x = int(x * self.width / self.pitch_config.width)
                        img_y = int(y * self.height / self.pitch_config.height)
                        
                        # Add ID text
                        cv2.putText(pitch_view, f"{int(player_id)}", (img_x - 5, img_y + 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Process ball if available
        if len(ball_detections) > 0:
            # Get ball position
            ball_position = ball_detections.get_anchors_coordinates(Position.CENTER)[0]
            
            # Transform to pitch coordinates
            ball_pitch_position = transformer.transform_points(np.array([ball_position]))[0]
            
            # Add to trail
            self.ball_trail.append(ball_pitch_position)
            if len(self.ball_trail) > self.max_trail_length:
                self.ball_trail.pop(0)
            
            # Draw ball trail
            if len(self.ball_trail) > 1:
                trail_array = np.array(self.ball_trail)
                pitch_view = draw_paths_on_pitch(
                    config=self.pitch_config,
                    paths=[trail_array],
                    color=self.ball_color,
                    width=2,
                    pitch=pitch_view
                )
            
            # Draw current ball position
            pitch_view = draw_points_on_pitch(
                config=self.pitch_config,
                xy=np.array([ball_pitch_position]),
                face_color=self.ball_color,
                edge_color=sv.Color.BLACK,
                radius=8,
                pitch=pitch_view
            )
        
        # Highlight player with possession
        if possession_info is not None and possession_info.get('player_id') is not None:
            player_id = possession_info['player_id']
            
            # Find player with possession
            for i, track_id in enumerate(person_detections.tracker_id):
                if int(track_id) == player_id:
                    # Get position
                    player_position = pitch_positions[i]
                    
                    # Draw possession highlight (larger circle)
                    pitch_view = draw_points_on_pitch(
                        config=self.pitch_config,
                        xy=np.array([player_position]),
                        face_color=self.possession_color,
                        edge_color=sv.Color.BLACK,
                        radius=18,
                        pitch=pitch_view
                    )
                    
                    # Add possession text
                    img_x = int(player_position[0] * self.width / self.pitch_config.width)
                    img_y = int(player_position[1] * self.height / self.pitch_config.height)
                    cv2.putText(pitch_view, "P", (img_x - 5, img_y - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.possession_color.as_bgr(), 2)
                    
                    break
        
        # Add info text
        cv2.putText(pitch_view, "Tactical View", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                   
        return pitch_view
