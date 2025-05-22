"""
Core processing module for Football AI application
Uses Supervision framework for detection, tracking, and annotation
"""

import cv2
import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO
from supervision.geometry.core import Position
from sports.configs.soccer import SoccerPitchConfiguration

try:
    from roboflow import Roboflow
except ImportError:
    print("Roboflow package not available, using Ultralytics models only")

class FootballProcessor:
    """Main processor for football video analysis."""
    
    def __init__(self, config, device='cuda'):
        """Initialize processor with models and components."""
        self.config = config
        self.device = device
        
        # Initialize all components in a structured way
        self._init_models()
        self._init_trackers()
        self._init_annotators()
        self._init_pitch_components()
        self._init_team_classifier()
        self._init_possession_detector()
    
    def _init_models(self):
        """Initialize detection and pose models with Ultralytics only."""
        print("Loading models...")
        
        # Use Ultralytics models directly
        model_path = self.config['models'].get('detection_backup', 'yolov8x.pt')
        self.model = YOLO(model_path)
        print(f"Loaded detection model: {model_path}")
        
        # Pose estimation model
        pose_path = self.config['models'].get('pose', 'yolov8x-pose.pt')
        self.pose_model = YOLO(pose_path)
        print(f"Loaded pose model: {pose_path}")
            
        # Segmentation model (SAM)
        if 'segmentation' in self.config['models']:
            try:
                from ultralytics import SAM
                self.sam_model = SAM(self.config['models']['segmentation'])
                print(f"Loaded SAM model: {self.config['models']['segmentation']}")
                self.enable_segmentation = True
            except Exception as e:
                print(f"Error loading SAM model: {e}")
                self.enable_segmentation = False
        else:
            self.enable_segmentation = False
        
        # Define class mappings
        self.player_classes = self.config['classes']['players']
        self.goalkeeper_classes = self.config['classes']['goalkeepers']
        self.referee_classes = self.config['classes']['referees']
        self.ball_classes = self.config['classes']['ball']
    
    def _init_trackers(self):
        """Initialize tracking and smoothing components."""
        # Create tracker based on configuration
        tracker_type = self.config['tracking'].get('tracker', 'bytetrack').lower()
        if tracker_type == "bytetrack":
            self.tracker = sv.ByteTrack()
        elif tracker_type == "botsort":
            self.tracker = sv.BotSORT()
        else:
            print(f"Unknown tracker type {tracker_type}, using ByteTrack")
            self.tracker = sv.ByteTrack()
            
        # Create detections smoother - check if it's available
        if hasattr(sv, 'DetectionsSmoother') and self.config['tracking'].get('smoothing_enabled', False):
            self.smoother = sv.DetectionsSmoother(
                length=self.config['tracking'].get('smoothing_length', 5)
            )
        else:
            # Fallback if not available
            print("DetectionsSmoother not available, disabling smoothing")
            self.smoother = None
            
        # Motion trail tracker - with proper initialization
        try:
            # Try with newer API
            self.trace_annotator = sv.TraceAnnotator()
        except TypeError:
            print("Warning: TraceAnnotator parameters not supported. Using defaults.")
            self.trace_annotator = sv.TraceAnnotator()
    
    def _init_annotators(self):
        """Initialize visualization annotators."""
        # Color palette for teams
        team_colors = [
            sv.Color.from_hex(self.config['visualization']['team_colors']['team_1']),
            sv.Color.from_hex(self.config['visualization']['team_colors']['team_2']),
            sv.Color.from_hex(self.config['visualization']['team_colors']['referee'])
        ]
        self.color_palette = sv.ColorPalette(colors=team_colors)
        
        # Box annotator
        self.box_annotator = sv.BoxAnnotator(
            color=self.color_palette,
            thickness=2
        )
        
        # Label annotator
        self.label_annotator = sv.LabelAnnotator(
            color=self.color_palette,
            text_color=sv.Color.from_hex(self.config['visualization']['label_color']),
            text_scale=0.5,
            text_thickness=1,
            text_position=Position.BOTTOM_CENTER
        )
        
        # Ball color for custom drawing
        self.ball_color = sv.Color.from_hex(self.config['visualization']['ball_color'])
        
        # Store colors in class for later use
        self.team_colors = {
            0: sv.Color.from_hex(self.config['visualization']['team_colors']['team_1']),
            1: sv.Color.from_hex(self.config['visualization']['team_colors']['team_2']),
            2: sv.Color.from_hex(self.config['visualization']['team_colors']['referee'])
        }
    
    def _init_pitch_components(self):
        """Initialize pitch visualization components."""
        # Soccer pitch configuration
        self.pitch_config = SoccerPitchConfiguration()
        self.pitch_width = self.pitch_config.width
        self.pitch_height = self.pitch_config.length
        
        # Set transformer to None (will be created during processing)
        self.transformer = None
        
        # Ball trail tracking
        self.ball_trail = []
        self.max_trail_length = 30
    
    def _init_team_classifier(self):
        """Initialize team classification."""
        from team_classifier import TeamClassifier
        
        self.team_classifier = TeamClassifier(
            model_path=self.config['models']['team_classifier'],
            token=self.config['api_keys']['huggingface_token']
        )
    
    def _init_possession_detector(self):
        """Initialize player possession detection."""
        from possession import PossessionDetector
        
        self.possession_detector = PossessionDetector(
            proximity_threshold=self.config['possession'].get('proximity_threshold', 50),
            possession_frames=self.config['possession'].get('possession_frames', 3),
            highlight_color=sv.Color.from_hex(self.config['possession'].get('highlight_color', '#00FF00'))
        )
    
    def process_frame(self, frame):
        """Process a single frame with the Football AI pipeline."""
        # Create output frame
        output_frame = frame.copy()
        
        # STEP 1: Run detection using Ultralytics directly
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # If no detections, return original frame
        if len(detections) == 0:
            return output_frame
        
        # STEP 2: Separate detections by class
        players_detections = self._filter_by_classes(detections, self.player_classes)
        goalkeeper_detections = self._filter_by_classes(detections, self.goalkeeper_classes)
        referee_detections = self._filter_by_classes(detections, self.referee_classes)
        ball_detections = self._filter_by_classes(detections, self.ball_classes)
        
        # Combine all person detections (players, goalkeepers, referees)
        person_detections = sv.Detections.merge([
            players_detections, 
            goalkeeper_detections, 
            referee_detections
        ]) if len(players_detections) > 0 or len(goalkeeper_detections) > 0 or len(referee_detections) > 0 else sv.Detections.empty()
        
        # STEP 3: Apply tracking to person detections
        if len(person_detections) > 0:
            person_detections = self.tracker.update_with_detections(person_detections)
            
            # Apply smoothing if enabled
            if self.smoother is not None:
                person_detections = self.smoother.update_with_detections(person_detections)
        
        # STEP 4: Create field transformation
        try:
            from sports.common.view import ViewTransformer
            
            # Create simple field points (fallback if detection fails)
            # These are approximations for a standard football field
            field_points = np.array([
                [100, 100],  # Top left
                [frame.shape[1] - 100, 100],  # Top right
                [frame.shape[1] - 100, frame.shape[0] - 100],  # Bottom right
                [100, frame.shape[0] - 100]  # Bottom left
            ])
            
            # Create matching pitch points
            pitch_points = np.array([
                [0, 0],  # Top left
                [self.pitch_width, 0],  # Top right
                [self.pitch_width, self.pitch_height],  # Bottom right
                [0, self.pitch_height]  # Bottom left
            ])
            
            # Create transformer
            self.transformer = ViewTransformer(
                source=field_points,
                target=pitch_points
            )
        except Exception as e:
            print(f"Error creating field transformation: {e}")
            self.transformer = None
        
        # STEP 5: Team classification
        if len(person_detections) > 0:
            # Extract crops for team classification
            person_crops = [self._crop_detection(frame, box) for box in person_detections.xyxy]
            team_ids = self.team_classifier.predict(person_crops)
            
            # Assign team IDs (0 or 1) to detections
            person_detections.class_id = np.array(team_ids)
        
        # STEP 6: Ball position and possession detection
        possession_info = None
        if len(ball_detections) > 0 and len(person_detections) > 0:
            # Get ball position
            ball_position = self._get_center_point(ball_detections.xyxy[0])
            
            # Update ball trail
            if self.transformer is not None:
                ball_pitch_position = self.transformer.transform_points(np.array([ball_position]))[0]
                self.ball_trail.append(ball_pitch_position)
                if len(self.ball_trail) > self.max_trail_length:
                    self.ball_trail.pop(0)
            
            # Detect possession
            possession_info = self.possession_detector.update(
                person_detections, 
                ball_position,
                transformer=self.transformer
            )
        
        # STEP 7: Run pose estimation on each person detection
        poses = None
        if hasattr(self, 'pose_model') and len(person_detections) > 0:
            poses = []
            for box in person_detections.xyxy:
                # Extract box coordinates
                x1, y1, x2, y2 = map(int, box)
                
                # Apply padding (10%)
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                
                # Ensure valid box with padding
                x1_pad = max(0, x1 - pad_x)
                y1_pad = max(0, y1 - pad_y)
                x2_pad = min(frame.shape[1], x2 + pad_x)
                y2_pad = min(frame.shape[0], y2 + pad_y)
                
                # Extract crop
                crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if crop.size == 0:
                    poses.append(None)
                    continue
                
                # Run pose detection on crop
                try:
                    pose_results = self.pose_model(crop, verbose=False)[0]
                    
                    # Check if pose detected
                    if (hasattr(pose_results, 'keypoints') and 
                        pose_results.keypoints is not None and 
                        len(pose_results.keypoints.data) > 0):
                        
                        # Get first pose (assuming only one person in crop)
                        keypoints = pose_results.keypoints.data[0].cpu().numpy()
                        
                        # Create pose data
                        pose = {
                            'keypoints': keypoints[:, :2].copy(),
                            'confidence': keypoints[:, 2].copy()
                        }
                        
                        # Adjust coordinates back to full frame
                        pose['keypoints'][:, 0] += x1_pad
                        pose['keypoints'][:, 1] += y1_pad
                        
                        poses.append(pose)
                    else:
                        poses.append(None)
                except Exception as e:
                    print(f"Error in pose estimation: {e}")
                    poses.append(None)
        
        # STEP 8: Run segmentation on each person detection
        segmentations = None
        if self.enable_segmentation and hasattr(self, 'sam_model') and len(person_detections) > 0:
            segmentations = []
            for box in person_detections.xyxy:
                # Apply padding (10%)
                x1, y1, x2, y2 = map(int, box)
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                
                # Ensure valid box with padding
                x1_pad = max(0, x1 - pad_x)
                y1_pad = max(0, y1 - pad_y)
                x2_pad = min(frame.shape[1], x2 + pad_x)
                y2_pad = min(frame.shape[0], y2 + pad_y)
                
                try:
                    # Run SAM with box prompt
                    sam_results = self.sam_model(
                        frame, 
                        bboxes=[box.tolist()],
                        verbose=False,
                    )[0]
                    
                    # Check if we have masks
                    if (hasattr(sam_results, 'masks') and 
                        sam_results.masks is not None and 
                        len(sam_results.masks.data) > 0):
                        
                        # Get first mask
                        mask = sam_results.masks.data[0].cpu().numpy()
                        segmentations.append(mask)
                    else:
                        segmentations.append(None)
                except Exception as e:
                    print(f"Error in segmentation: {e}")
                    segmentations.append(None)
        
        # STEP 9: Create visualization
        # Draw segmentations if available
        if segmentations is not None:
            for i, mask in enumerate(segmentations):
                if mask is None or i >= len(person_detections.class_id):
                    continue
                    
                # Get team color
                team_id = person_detections.class_id[i]
                if team_id in self.team_colors:
                    color = self.team_colors[team_id].as_bgr()
                else:
                    color = (0, 255, 0)  # Default to green
                
                # Apply mask with team color
                alpha = 0.5
                overlay = output_frame.copy()
                overlay[mask > 0.5] = color
                output_frame = cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0)
        
        # Draw ball detections
        if len(ball_detections) > 0:
            for box in ball_detections.xyxy:
                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius = max(5, (x2 - x1) // 4)
                cv2.circle(output_frame, (center_x, center_y), radius, self.ball_color.as_bgr(), -1)
                cv2.circle(output_frame, (center_x, center_y), radius, (0, 0, 0), 1)  # Black outline
        
        # Create labels with team and ID information
        if len(person_detections) > 0:
            # Create labels
            labels = [f"#{track_id}" for track_id in person_detections.tracker_id]
            
            # Draw boxes with team colors
            output_frame = self.box_annotator.annotate(output_frame, person_detections)
            
            # Draw labels
            output_frame = self.label_annotator.annotate(
                output_frame, person_detections, labels=labels
            )
            
            # Add motion traces
            output_frame = self.trace_annotator.annotate(output_frame, person_detections)
        
        # Draw poses if available
        if poses is not None:
            for person_idx, pose in enumerate(poses):
                if pose is not None and person_idx < len(person_detections.class_id):
                    team_id = person_detections.class_id[person_idx]
                    color = self.team_colors[team_id].as_bgr()
                    output_frame = self._draw_pose(output_frame, pose, color)
        
        # Highlight player with possession
        if possession_info is not None and possession_info.get('player_id') is not None:
            output_frame = self.possession_detector.highlight_possession(
                output_frame, 
                person_detections,
                possession_info
            )
        
        # STEP 10: Render tactical view if transformer is available
        if self.transformer is not None:
            # Create tactical view
            pitch_view = self._render_tactical_view(
                person_detections=person_detections,
                ball_detections=ball_detections,
                possession_info=possession_info
            )
            
            # Combine views side by side
            output_frame = self._combine_views(output_frame, pitch_view)
        
        return output_frame
    
    def _filter_by_classes(self, detections, class_ids):
        """Filter detections by class IDs."""
        if len(detections) == 0:
            return sv.Detections.empty()
            
        mask = np.isin(detections.class_id, class_ids)
        return detections[mask]
    
    def _crop_detection(self, frame, bbox):
        """Extract crop from frame based on bounding box."""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure valid coordinates
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            # Return empty image if invalid
            return np.zeros((10, 10, 3), dtype=np.uint8)
            
        return frame[y1:y2, x1:x2]
    
    def _get_center_point(self, bbox):
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    
    def _draw_pose(self, frame, pose, color=(0, 255, 0), thickness=2):
        """Draw pose keypoints and connections on frame."""
        if pose is None or 'keypoints' not in pose:
            return frame
            
        keypoints = pose['keypoints']
        confidence = pose['confidence']
        
        # Draw keypoints
        for i, (x, y) in enumerate(keypoints):
            if confidence[i] > 0.5:
                cv2.circle(frame, (int(x), int(y)), 5, color, -1)
        
        # Draw connections (simplified COCO connections)
        connections = [
            # Limbs
            (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
            # Body
            (5, 6), (5, 11), (6, 12), (11, 12)
        ]
        
        for p1, p2 in connections:
            if p1 < len(keypoints) and p2 < len(keypoints) and confidence[p1] > 0.5 and confidence[p2] > 0.5:
                pt1 = (int(keypoints[p1][0]), int(keypoints[p1][1]))
                pt2 = (int(keypoints[p2][0]), int(keypoints[p2][1]))
                cv2.line(frame, pt1, pt2, color, thickness)
        
        return frame
    
    def _render_tactical_view(self, person_detections, ball_detections, possession_info=None):
        """Render tactical pitch view."""
        # Create pitch image
        pitch_width = self.config['visualization'].get('pitch_view_width', 600)
        pitch_height = self.config['visualization'].get('pitch_view_height', 800)
        
        # Create blank pitch image
        pitch_view = np.ones((pitch_height, pitch_width, 3), dtype=np.uint8) * 255
        
        # Draw pitch lines (simplified)
        # Field outline
        cv2.rectangle(pitch_view, (50, 50), (pitch_width-50, pitch_height-50), (0, 128, 0), 2)
        
        # Center line
        cv2.line(pitch_view, (50, pitch_height//2), (pitch_width-50, pitch_height//2), (0, 128, 0), 2)
        
        # Center circle
        cv2.circle(pitch_view, (pitch_width//2, pitch_height//2), 60, (0, 128, 0), 2)
        
        # Penalty areas
        cv2.rectangle(pitch_view, (50, 50), (200, 200), (0, 128, 0), 2)  # Top penalty area
        cv2.rectangle(pitch_view, (50, pitch_height-200), (200, pitch_height-50), (0, 128, 0), 2)  # Bottom penalty area
        
        # Draw players if available
        if len(person_detections) > 0 and self.transformer is not None:
            # Get player positions in frame coordinates
            player_positions = person_detections.get_anchors_coordinates(Position.BOTTOM_CENTER)
            
            # Transform to pitch coordinates
            pitch_positions = self.transformer.transform_points(player_positions)
            
            # Draw players by team
            for i, (pos, team_id) in enumerate(zip(pitch_positions, person_detections.class_id)):
                # Scale to pitch view
                x = int(pos[0] * pitch_width / self.pitch_width)
                y = int(pos[1] * pitch_height / self.pitch_height)
                
                # Get color
                color = self.team_colors.get(team_id, sv.Color.WHITE).as_bgr()
                
                # Draw player
                cv2.circle(pitch_view, (x, y), 10, color, -1)
                cv2.circle(pitch_view, (x, y), 10, (0, 0, 0), 1)  # Black outline
                
                # Add player ID
                if hasattr(person_detections, 'tracker_id') and i < len(person_detections.tracker_id):
                    player_id = person_detections.tracker_id[i]
                    cv2.putText(pitch_view, f"{player_id}", (x-5, y+5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                
                # Highlight player with possession
                if (possession_info is not None and 
                    possession_info.get('player_id') is not None and 
                    hasattr(person_detections, 'tracker_id') and
                    i < len(person_detections.tracker_id) and
                    person_detections.tracker_id[i] == possession_info['player_id']):
                    
                    cv2.circle(pitch_view, (x, y), 15, (0, 255, 0), 2)  # Green circle
                    cv2.putText(pitch_view, "P", (x-5, y-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw ball and trail
        if len(ball_detections) > 0 and self.transformer is not None:
            # Get ball position
            ball_position = ball_detections.get_anchors_coordinates(Position.CENTER)[0]
            
            # Transform to pitch coordinates
            ball_pitch_position = self.transformer.transform_points(np.array([ball_position]))[0]
            
            # Scale to pitch view
            ball_x = int(ball_pitch_position[0] * pitch_width / self.pitch_width)
            ball_y = int(ball_pitch_position[1] * pitch_height / self.pitch_height)
            
            # Draw ball trail
            if len(self.ball_trail) > 1:
                for i in range(1, len(self.ball_trail)):
                    prev_x = int(self.ball_trail[i-1][0] * pitch_width / self.pitch_width)
                    prev_y = int(self.ball_trail[i-1][1] * pitch_height / self.pitch_height)
                    curr_x = int(self.ball_trail[i][0] * pitch_width / self.pitch_width)
                    curr_y = int(self.ball_trail[i][1] * pitch_height / self.pitch_height)
                    
                    # Thicker/more opaque for more recent positions
                    alpha = 0.5 + 0.5 * i / len(self.ball_trail)
                    thickness = max(1, int(2 * i / len(self.ball_trail)))
                    cv2.line(pitch_view, (prev_x, prev_y), (curr_x, curr_y), 
                            self.ball_color.as_bgr(), thickness)
            
            # Draw current ball position
            cv2.circle(pitch_view, (ball_x, ball_y), 8, self.ball_color.as_bgr(), -1)
            cv2.circle(pitch_view, (ball_x, ball_y), 8, (0, 0, 0), 1)  # Black outline
        
        # Add title
        cv2.putText(pitch_view, "Tactical View", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                   
        return pitch_view
    
    def _combine_views(self, main_view, pitch_view):
        """Combine main view and pitch view side by side."""
        # Resize pitch view to target height
        main_h, main_w = main_view.shape[:2]
        pitch_h, pitch_w = pitch_view.shape[:2]
        
        # Target height should match the main view
        target_h = main_h
        target_w = int(pitch_w * (target_h / pitch_h))
        
        # Resize pitch view
        pitch_view_resized = cv2.resize(pitch_view, (target_w, target_h))
        
        # Create combined view
        combined_w = main_w + target_w
        combined_view = np.zeros((target_h, combined_w, 3), dtype=np.uint8)
        
        # Add views
        combined_view[:, :main_w] = main_view
        combined_view[:, main_w:] = pitch_view_resized
        
        # Add labels
        cv2.putText(combined_view, "Live View", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_view, "Tactical View", (main_w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return combined_view
