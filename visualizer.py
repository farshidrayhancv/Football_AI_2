"""
Football AI V2 - Visualization using Supervision
Handles all visual annotations and tactical view
"""

import cv2
import numpy as np
import supervision as sv
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch, draw_paths_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration


class FootballVisualizer:
    """Handles all visualization using supervision library."""
    
    def __init__(self, config):
        """Initialize visualizer with configuration."""
        self.config = config
        
        # Initialize colors
        self.team_colors = {
            0: sv.Color.from_hex(config['display']['team_colors']['team_1']),
            1: sv.Color.from_hex(config['display']['team_colors']['team_2'])
        }
        self.referee_color = sv.Color.from_hex(config['display']['referee_color'])
        self.ball_color = sv.Color.from_hex(config['display']['ball_color'])
        
        # Initialize annotators
        self._init_annotators()
        
        # Pitch configuration
        self.pitch_config = SoccerPitchConfiguration()
        
        print("✓ Visualizer initialized")
    
    def _init_annotators(self):
        """Initialize supervision annotators."""
        # Color palette for teams and referees
        # Index 0 = Team 1, Index 1 = Team 2, Index 2 = Referee
        colors = [
            self.config['display']['team_colors']['team_1'],
            self.config['display']['team_colors']['team_2'], 
            self.config['display']['referee_color']
        ]
        
        # Bounding box annotator
        self.player_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(colors),
            thickness=2
        )
        
        # Label annotator
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(colors),
            text_color=sv.Color.BLACK,
            text_scale=0.5,
            text_thickness=1
        )
        
        # Triangle annotator for ball
        self.triangle_annotator = sv.TriangleAnnotator(
            color=self.ball_color,
            base=20,
            height=15
        )
    
    def annotate_frame(self, frame, detections, poses=None, segments=None, possession_info=None):
        """Annotate frame with all detections and features."""
        annotated = frame.copy()
        
        # Draw segmentations first (as background)
        if segments and self.config['display'].get('show_segmentation', True):
            annotated = self._draw_segmentations(annotated, detections, segments)
        
        # Draw poses
        if poses and self.config['display'].get('show_pose', True):
            annotated = self._draw_poses(annotated, detections, poses)
        
        # Draw bounding boxes for each category separately to ensure proper team coloring
        for category in ['players', 'goalkeepers', 'referees']:
            if len(detections[category]) > 0:
                category_detections = detections[category]
                
                # Draw bounding boxes
                annotated = self.player_annotator.annotate(annotated, category_detections)
                
                # Draw labels with tracking IDs
                if category_detections.tracker_id is not None:
                    labels = [f"#{track_id}" for track_id in category_detections.tracker_id]
                    annotated = self.label_annotator.annotate(annotated, category_detections, labels=labels)
        
        # Draw ball
        if len(detections['ball']) > 0:
            annotated = self.triangle_annotator.annotate(annotated, detections['ball'])
        
        # Highlight possession
        if possession_info and self.config['possession_detection'].get('enable', True):
            annotated = self._highlight_possession(annotated, detections, possession_info)
        
        return annotated
    
    def _draw_segmentations(self, frame, detections, segments):
        """Draw segmentation masks with team colors."""
        overlay = frame.copy()
        alpha = self.config['display'].get('segmentation_alpha', 0.6)
        
        # Draw player segments
        for category in ['players', 'goalkeepers', 'referees']:
            if category not in segments or not segments[category]:
                continue
            
            category_detections = detections[category]
            category_segments = segments[category]
            
            for i, mask in enumerate(category_segments):
                if mask is None or i >= len(category_detections):
                    continue
                
                # Get color based on team classification
                if category == 'referees':
                    color = self.referee_color.as_bgr()
                else:
                    # Use the team classification from class_id (0 = team_1, 1 = team_2)
                    if (hasattr(category_detections, 'class_id') and 
                        category_detections.class_id is not None and 
                        i < len(category_detections.class_id)):
                        team_id = category_detections.class_id[i]
                        if team_id in self.team_colors:
                            color = self.team_colors[team_id].as_bgr()
                        else:
                            color = self.team_colors[0].as_bgr()  # Default to team 1
                    else:
                        color = self.team_colors[0].as_bgr()  # Default to team 1
                
                # Apply mask
                mask_bool = mask > 0.5
                overlay[mask_bool] = color
        
        # Blend with original frame
        frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        return frame
    
    def _draw_poses(self, frame, detections, poses):
        """Draw pose keypoints and skeleton."""
        # COCO pose connections
        connections = [
            (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
            (5, 6), (5, 11), (6, 12), (11, 12)  # Body
        ]
        
        for category in ['players', 'goalkeepers', 'referees']:
            if category not in poses or not poses[category]:
                continue
            
            category_detections = detections[category]
            category_poses = poses[category]
            
            for i, pose in enumerate(category_poses):
                if pose is None or i >= len(category_detections):
                    continue
                
                # Get color based on team classification
                if category == 'referees':
                    color = self.referee_color.as_bgr()
                else:
                    # Use the team classification from class_id (0 = team_1, 1 = team_2)
                    if (hasattr(category_detections, 'class_id') and 
                        category_detections.class_id is not None and 
                        i < len(category_detections.class_id)):
                        team_id = category_detections.class_id[i]
                        if team_id in self.team_colors:
                            color = self.team_colors[team_id].as_bgr()
                        else:
                            color = self.team_colors[0].as_bgr()  # Default to team 1
                    else:
                        color = self.team_colors[0].as_bgr()  # Default to team 1
                
                keypoints = pose['keypoints']
                confidence = pose['confidence']
                
                # Draw keypoints
                for j, (x, y) in enumerate(keypoints):
                    if confidence[j] > 0.5:
                        cv2.circle(frame, (int(x), int(y)), 4, color, -1)
                
                # Draw skeleton
                for p1, p2 in connections:
                    if (p1 < len(keypoints) and p2 < len(keypoints) and 
                        confidence[p1] > 0.5 and confidence[p2] > 0.5):
                        pt1 = (int(keypoints[p1][0]), int(keypoints[p1][1]))
                        pt2 = (int(keypoints[p2][0]), int(keypoints[p2][1]))
                        cv2.line(frame, pt1, pt2, color, 2)
        
        return frame
    
    def _highlight_possession(self, frame, detections, possession_info):
        """Highlight player with possession."""
        if not possession_info or 'player_id' not in possession_info:
            return frame
        
        player_id = possession_info['player_id']
        category = possession_info.get('category', 'players')
        
        # Find player in detections
        if category not in detections or len(detections[category]) == 0:
            return frame
        
        category_detections = detections[category]
        if not hasattr(category_detections, 'tracker_id') or category_detections.tracker_id is None:
            return frame
        
        # Find matching player
        for i, track_id in enumerate(category_detections.tracker_id):
            if int(track_id) == player_id:
                # Get bounding box
                box = category_detections.xyxy[i].astype(int)
                x1, y1, x2, y2 = box
                
                # Draw thick highlight box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)
                
                # Draw possession indicator
                center_x = (x1 + x2) // 2
                center_y = y1 - 30
                cv2.circle(frame, (center_x, center_y), 15, (0, 255, 255), -1)
                cv2.putText(frame, "P", (center_x - 5, center_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Add possession text
                cv2.putText(frame, "POSSESSION", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                break
        
        return frame
    
    def _merge_human_detections(self, detections):
        """Merge human detections for unified processing."""
        human_dets = []
        
        for category in ['players', 'goalkeepers', 'referees']:
            if len(detections[category]) > 0:
                human_dets.append(detections[category])
        
        if human_dets:
            return sv.Detections.merge(human_dets)
        else:
            return sv.Detections.empty()
    
    def create_tactical_view(self, detections, transformer, ball_trail, possession_info=None):
        """Create tactical pitch view."""
        # Create pitch background
        pitch_view = draw_pitch(self.pitch_config)
        
        if transformer is None:
            return pitch_view
        
        # Draw players
        pitch_view = self._draw_players_on_pitch(pitch_view, detections, transformer, possession_info)
        
        # Draw ball trail
        if ball_trail and len(ball_trail) > 1:
            trail_array = np.array(ball_trail)
            pitch_view = draw_paths_on_pitch(
                config=self.pitch_config,
                paths=[trail_array],
                color=self.ball_color,
                pitch=pitch_view
            )
        
        # Draw current ball position
        if len(detections['ball']) > 0:
            ball_pos = detections['ball'].get_anchors_coordinates(sv.Position.CENTER)
            if len(ball_pos) > 0:
                pitch_ball_pos = transformer.transform_points(ball_pos)
                if len(pitch_ball_pos) > 0:
                    pitch_view = draw_points_on_pitch(
                        config=self.pitch_config,
                        xy=pitch_ball_pos,
                        face_color=self.ball_color,
                        edge_color=sv.Color.BLACK,
                        radius=8,
                        pitch=pitch_view
                    )
        
        return pitch_view
    
    def _draw_players_on_pitch(self, pitch_view, detections, transformer, possession_info):
        """Draw players on tactical pitch."""
        for category in ['players', 'goalkeepers', 'referees']:
            if len(detections[category]) == 0:
                continue
            
            # Get positions
            positions = detections[category].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_positions = transformer.transform_points(positions)
            
            if len(pitch_positions) == 0:
                continue
            
            category_detections = detections[category]
            
            if category == 'referees':
                # Draw all referees with referee color
                pitch_view = draw_points_on_pitch(
                    config=self.pitch_config,
                    xy=pitch_positions,
                    face_color=self.referee_color,
                    edge_color=sv.Color.BLACK,
                    radius=12,
                    pitch=pitch_view
                )
            else:
                # Draw players/goalkeepers by team using team classification
                if hasattr(category_detections, 'class_id') and category_detections.class_id is not None:
                    for team_id in [0, 1]:
                        team_mask = category_detections.class_id == team_id
                        if np.any(team_mask):
                            team_positions = pitch_positions[team_mask]
                            if team_id in self.team_colors:
                                color = self.team_colors[team_id]
                            else:
                                color = self.team_colors[0]  # Default to team 1
                            
                            pitch_view = draw_points_on_pitch(
                                config=self.pitch_config,
                                xy=team_positions,
                                face_color=color,
                                edge_color=sv.Color.BLACK,
                                radius=12,
                                pitch=pitch_view
                            )
                else:
                    # Fallback: draw all with team 1 color if no team classification
                    pitch_view = draw_points_on_pitch(
                        config=self.pitch_config,
                        xy=pitch_positions,
                        face_color=self.team_colors[0],
                        edge_color=sv.Color.BLACK,
                        radius=12,
                        pitch=pitch_view
                    )
        
        # Highlight possession
        if possession_info and 'player_id' in possession_info:
            self._highlight_possession_on_pitch(pitch_view, detections, transformer, possession_info)
        
        return pitch_view
    
    def _highlight_possession_on_pitch(self, pitch_view, detections, transformer, possession_info):
        """Highlight player with possession on tactical view."""
        player_id = possession_info['player_id']
        category = possession_info.get('category', 'players')
        
        if category not in detections or len(detections[category]) == 0:
            return
        
        category_detections = detections[category]
        if not hasattr(category_detections, 'tracker_id') or category_detections.tracker_id is None:
            return
        
        # Find player
        for i, track_id in enumerate(category_detections.tracker_id):
            if int(track_id) == player_id:
                # Get position
                position = category_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)[i:i+1]
                pitch_position = transformer.transform_points(position)
                
                if len(pitch_position) > 0:
                    # Draw possession highlight
                    draw_points_on_pitch(
                        config=self.pitch_config,
                        xy=pitch_position,
                        face_color=sv.Color.from_hex('#00FF00'),  # Green
                        edge_color=sv.Color.BLACK,
                        radius=18,
                        pitch=pitch_view
                    )
                break
    
    def combine_views(self, main_view, tactical_view):
        """Combine main view and tactical view side by side."""
        main_h, main_w = main_view.shape[:2]
        tactical_h, tactical_w = tactical_view.shape[:2]
        
        # Resize tactical view to match main view height
        target_w = int(tactical_w * (main_h / tactical_h))
        tactical_resized = cv2.resize(tactical_view, (target_w, main_h))
        
        # Create combined view
        combined_w = main_w + target_w
        combined = np.zeros((main_h, combined_w, 3), dtype=np.uint8)
        
        # Place views
        combined[:, :main_w] = main_view
        combined[:, main_w:] = tactical_resized
        
        # Add labels
        cv2.putText(combined, "Live View", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Tactical View", (main_w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return combined