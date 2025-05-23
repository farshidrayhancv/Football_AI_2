"""
Football AI V2 - Simple Multi-Resolution Core Processing Logic
Different resolutions for different models to manage GPU memory and performance
"""

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
import os

from models import FootballModels
from visualizer import FootballVisualizer


class FootballProcessor:
    """Main processor with simple per-model resolution control."""
    
    def __init__(self, config, device='cuda'):
        """Initialize processor."""
        self.config = config
        self.device = device
        
        # Get resolution settings
        self.resolutions = config.get('resolution', {})
        self.object_detection_res = self.resolutions.get('object_detection', [640, 640])
        self.pose_estimation_res = self.resolutions.get('pose_estimation', [480, 480])
        self.segmentation_res = self.resolutions.get('segmentation', [512, 512])
        self.field_detection_res = self.resolutions.get('field_detection', [1280, 720])
        self.output_res = self.resolutions.get('output', [800, 600])
        
        print(f"✓ Object Detection Resolution: {self.object_detection_res[0]}x{self.object_detection_res[1]}")
        print(f"✓ Pose Estimation Resolution: {self.pose_estimation_res[0]}x{self.pose_estimation_res[1]}")
        print(f"✓ Segmentation Resolution: {self.segmentation_res[0]}x{self.segmentation_res[1]}")
        print(f"✓ Field Detection Resolution: {self.field_detection_res[0]}x{self.field_detection_res[1]}")
        print(f"✓ Output Resolution: {self.output_res[0]}x{self.output_res[1]}")
        
        # Initialize models
        self.models = FootballModels(config, device)
        
        # Initialize visualizer
        self.visualizer = FootballVisualizer(config)
        
        # Initialize tracker
        self.tracker = sv.ByteTrack(track_activation_threshold=0.8)
        self.smoother = sv.DetectionsSmoother(length=5)
        
        # Ball trail for tactical view
        self.ball_trail = []
        self.max_trail_length = 30
        
        # Current transformation matrix
        self.transformer = None
        
        print("Football AI V2 processor initialized successfully!")
    
    def _resize_frame(self, frame, target_resolution):
        """Resize frame to target resolution."""
        if target_resolution is None:
            return frame, (1.0, 1.0)
        
        original_height, original_width = frame.shape[:2]
        target_width, target_height = target_resolution
        
        # Calculate scale factors
        scale_x = original_width / target_width
        scale_y = original_height / target_height
        
        # Resize frame
        resized_frame = cv2.resize(frame, (target_width, target_height))
        
        return resized_frame, (scale_x, scale_y)
    
    def _scale_detections(self, detections, scale_factors):
        """Scale detections back to original resolution."""
        if scale_factors == (1.0, 1.0) or len(detections) == 0:
            return detections
        
        scale_x, scale_y = scale_factors
        
        # Create scaled detections
        scaled_detections = sv.Detections(
            xyxy=detections.xyxy.copy(),
            confidence=detections.confidence.copy() if detections.confidence is not None else None,
            class_id=detections.class_id.copy() if detections.class_id is not None else None,
            tracker_id=detections.tracker_id.copy() if detections.tracker_id is not None else None
        )
        
        # Scale bounding boxes
        scaled_detections.xyxy[:, [0, 2]] *= scale_x  # x coordinates
        scaled_detections.xyxy[:, [1, 3]] *= scale_y  # y coordinates
        
        return scaled_detections
    
    def _scale_poses(self, poses, scale_factors):
        """Scale pose keypoints back to original resolution."""
        if scale_factors == (1.0, 1.0) or not poses:
            return poses
        
        scale_x, scale_y = scale_factors
        scaled_poses = []
        
        for pose in poses:
            if pose is not None:
                scaled_pose = pose.copy()
                # Scale keypoints
                scaled_pose['keypoints'][:, 0] *= scale_x
                scaled_pose['keypoints'][:, 1] *= scale_y
                scaled_poses.append(scaled_pose)
            else:
                scaled_poses.append(None)
        
        return scaled_poses
    
    def _scale_segments(self, segments, original_shape):
        """Scale segmentation masks back to original resolution."""
        if not segments:
            return segments
        
        original_height, original_width = original_shape[:2]
        scaled_segments = []
        
        for mask in segments:
            if mask is not None and isinstance(mask, np.ndarray):
                # Resize mask to original dimensions
                scaled_mask = cv2.resize(mask.astype(np.uint8), 
                                       (original_width, original_height),
                                       interpolation=cv2.INTER_NEAREST)
                scaled_segments.append(scaled_mask.astype(bool))
            else:
                scaled_segments.append(None)
        
        return scaled_segments
    
    def process_video(self):
        """Process the entire video."""
        input_path = self.config['video']['input_path']
        output_path = self.config['video']['output_path']
        
        # Check input file
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {original_width}x{original_height}, {fps} fps, {total_frames} frames")
        
        # Create video writer with configured output resolution
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_width = self.output_res[0]
        out_height = self.output_res[1]
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        # Train team classifier on first few frames
        self._train_team_classifier(cap, original_width, original_height)
        
        # Reset video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Process frames
        frame_count = 0
        stride = self.config['video'].get('stride', 1)
        
        with tqdm(total=total_frames//stride, desc="Processing") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on stride
                if frame_count % stride != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                result_frame = self.process_frame(frame)
                
                # Resize to output resolution
                if result_frame.shape[:2] != (out_height, out_width):
                    result_frame = cv2.resize(result_frame, (out_width, out_height))
                
                # Write frame
                out.write(result_frame)
                
                frame_count += 1
                pbar.update(1)
                
                # Save preview occasionally
                if frame_count % 500 == 0:
                    preview_path = output_path.replace('.mp4', f'_preview_{frame_count}.jpg')
                    cv2.imwrite(preview_path, result_frame)
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"Processing complete! Output saved to: {output_path}")
    
    def _train_team_classifier(self, cap, original_width, original_height):
        """Train team classifier on sample frames."""
        print("Training team classifier...")
        
        crops = []
        sample_frames = 50
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // sample_frames)
        
        for i in range(sample_frames):
            # Seek to frame
            frame_pos = i * frame_interval
            if frame_pos >= total_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for object detection
            detection_frame, scale_factors = self._resize_frame(frame, self.object_detection_res)
            
            # Detect players only
            detections = self.models.detect_objects(detection_frame)
            player_detections = detections['players']
            
            # Extract crops and scale back to original if needed
            if len(player_detections) > 0:
                try:
                    # Scale detections back to original frame
                    scaled_players = self._scale_detections(player_detections, scale_factors)
                    player_crops = [sv.crop_image(frame, xyxy) for xyxy in scaled_players.xyxy]
                    
                    # Filter out empty crops
                    valid_crops = [crop for crop in player_crops if crop.size > 0]
                    crops.extend(valid_crops)
                    
                except Exception as e:
                    print(f"Error extracting crops from frame {frame_pos}: {e}")
        
        # Train classifier
        if crops:
            self.models.train_team_classifier(crops)
            print(f"Team classifier trained on {len(crops)} player crops from {sample_frames} frames")
        else:
            print("Warning: No player crops found for team classification")
    
    def process_frame(self, frame):
        """Process a single frame using different resolutions for different models."""
        original_shape = frame.shape
        
        # Step 1: Object detection at configured resolution
        detection_frame, detection_scale = self._resize_frame(frame, self.object_detection_res)
        detections = self.models.detect_objects(detection_frame)
        
        # Scale detections back to original resolution
        for category in ['players', 'goalkeepers', 'referees', 'ball']:
            if len(detections[category]) > 0:
                detections[category] = self._scale_detections(detections[category], detection_scale)
        
        # Step 2: Field detection at configured resolution (usually higher for accuracy)
        field_frame, field_scale = self._resize_frame(frame, self.field_detection_res)
        field_keypoints = self.models.detect_field(field_frame)
        
        # Scale field keypoints back if needed
        if field_keypoints is not None and field_scale != (1.0, 1.0):
            # Scale keypoints back to original resolution
            if len(field_keypoints.xy) > 0:
                field_keypoints.xy[0][:, 0] *= field_scale[0]
                field_keypoints.xy[0][:, 1] *= field_scale[1]
        
        self.transformer = self.models.create_transformer(field_keypoints)
        
        # Step 3: Track objects
        human_detections = self._merge_human_detections(detections)
        if len(human_detections) > 0:
            human_detections = self.tracker.update_with_detections(human_detections)
            human_detections = self.smoother.update_with_detections(human_detections)
            
            # Split back into categories
            detections = self._split_human_detections(human_detections, detections)
        
        # Step 4: Team classification (use original frame for best quality)
        if len(detections['players']) > 0:
            team_ids = self.models.classify_teams(frame, detections['players'])
            detections['players'].class_id = np.array(team_ids)
        
        # Assign goalkeeper teams
        if len(detections['goalkeepers']) > 0 and len(detections['players']) > 0:
            gk_team_ids = self._resolve_goalkeeper_teams(detections['players'], detections['goalkeepers'])
            detections['goalkeepers'].class_id = np.array(gk_team_ids)
        elif len(detections['goalkeepers']) > 0:
            gk_team_ids = self.models.classify_teams(frame, detections['goalkeepers'])
            detections['goalkeepers'].class_id = np.array(gk_team_ids)
        
        # Referees get special class_id
        if len(detections['referees']) > 0:
            detections['referees'].class_id = np.full(len(detections['referees']), 2, dtype=int)
        
        # Step 5: Pose estimation at configured resolution
        poses = {}
        if len(human_detections) > 0:
            # Resize frame for pose estimation
            pose_frame, pose_scale = self._resize_frame(frame, self.pose_estimation_res)
            
            # Scale detections down for pose estimation
            pose_detections = self._scale_detections(human_detections, (1/pose_scale[0], 1/pose_scale[1]))
            
            all_poses = self.models.estimate_poses(pose_frame, pose_detections)
            
            # Scale poses back to original resolution
            all_poses = self._scale_poses(all_poses, pose_scale)
            
            # Split poses back to categories
            poses = self._split_poses_by_category(all_poses, detections)
        
        # Step 6: Segmentation at configured resolution
        segments = {}
        if len(human_detections) > 0:
            # Resize frame for segmentation
            seg_frame, seg_scale = self._resize_frame(frame, self.segmentation_res)
            
            # Scale detections down for segmentation
            seg_detections = self._scale_detections(human_detections, (1/seg_scale[0], 1/seg_scale[1]))
            
            all_segments = self.models.segment_players(seg_frame, seg_detections)
            
            # Scale segments back to original resolution
            all_segments = self._scale_segments(all_segments, original_shape)
            
            # Split segments back to categories
            segments = self._split_segments_by_category(all_segments, detections)
        
        # Step 7: Possession detection
        possession_info = self._detect_possession(detections, self.transformer)
        
        # Step 8: Update ball trail
        self._update_ball_trail(detections['ball'], self.transformer)
        
        # Step 9: Create visualization
        annotated_frame = self.visualizer.annotate_frame(
            frame, detections, poses, segments, possession_info
        )
        
        # Step 10: Create tactical view
        tactical_view = self.visualizer.create_tactical_view(
            detections, self.transformer, self.ball_trail, possession_info
        )
        
        # Step 11: Combine views
        combined_frame = self.visualizer.combine_views(annotated_frame, tactical_view)
        
        return combined_frame
    
    # Helper methods (same as before)
    def _merge_human_detections(self, detections):
        """Merge player, goalkeeper, and referee detections for tracking."""
        human_dets = []
        
        for category in ['players', 'goalkeepers', 'referees']:
            if len(detections[category]) > 0:
                human_dets.append(detections[category])
        
        if human_dets:
            return sv.Detections.merge(human_dets)
        else:
            return sv.Detections.empty()
    
    def _split_human_detections(self, human_detections, original_detections):
        """Split tracked human detections back into categories."""
        if len(human_detections) == 0:
            return original_detections
        
        detections = {
            'players': human_detections[human_detections.class_id == 2],
            'goalkeepers': human_detections[human_detections.class_id == 1], 
            'referees': human_detections[human_detections.class_id == 3],
            'ball': original_detections['ball']
        }
        
        return detections
    
    def _detect_possession(self, detections, transformer):
        """Simple possession detection based on proximity to ball."""
        if len(detections['ball']) == 0 or transformer is None:
            return None
        
        ball_pos = detections['ball'].get_anchors_coordinates(sv.Position.CENTER)
        if len(ball_pos) == 0:
            return None
        
        ball_pos = ball_pos[0]
        
        min_distance = float('inf')
        closest_player = None
        
        for category in ['players', 'goalkeepers', 'referees']:
            if len(detections[category]) == 0:
                continue
            
            positions = detections[category].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            
            for i, pos in enumerate(positions):
                distance = np.linalg.norm(pos - ball_pos)
                if distance < min_distance and distance < 100:
                    min_distance = distance
                    if hasattr(detections[category], 'tracker_id') and detections[category].tracker_id is not None:
                        closest_player = {
                            'player_id': int(detections[category].tracker_id[i]),
                            'category': category,
                            'distance': distance
                        }
        
        return closest_player
    
    def _update_ball_trail(self, ball_detections, transformer):
        """Update ball trail for tactical view."""
        if len(ball_detections) == 0 or transformer is None:
            return
        
        ball_pos = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
        if len(ball_pos) > 0:
            pitch_pos = transformer.transform_points(ball_pos)
            if len(pitch_pos) > 0:
                self.ball_trail.append(pitch_pos[0])
                if len(self.ball_trail) > self.max_trail_length:
                    self.ball_trail.pop(0)
    
    def _split_poses_by_category(self, all_poses, detections):
        """Split poses back into detection categories."""
        poses = {'players': [], 'goalkeepers': [], 'referees': []}
        
        if not all_poses:
            return poses
        
        pose_idx = 0
        for category in ['players', 'goalkeepers', 'referees']:
            category_count = len(detections[category])
            poses[category] = all_poses[pose_idx:pose_idx + category_count]
            pose_idx += category_count
        
        return poses
    
    def _split_segments_by_category(self, all_segments, detections):
        """Split segments back into detection categories."""
        segments = {'players': [], 'goalkeepers': [], 'referees': []}
        
        if not all_segments:
            return segments
        
        segment_idx = 0
        for category in ['players', 'goalkeepers', 'referees']:
            category_count = len(detections[category])
            segments[category] = all_segments[segment_idx:segment_idx + category_count]
            segment_idx += category_count
        
        return segments
    
    def _resolve_goalkeeper_teams(self, players, goalkeepers):
        """Assign goalkeepers to teams based on proximity to player centroids."""
        if len(goalkeepers) == 0 or len(players) == 0:
            return np.zeros(len(goalkeepers), dtype=int)
        
        try:
            goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            
            if not hasattr(players, 'class_id') or players.class_id is None:
                return np.zeros(len(goalkeepers), dtype=int)
            
            team_0_players = players_xy[players.class_id == 0]
            team_1_players = players_xy[players.class_id == 1]
            
            if len(team_0_players) == 0 or len(team_1_players) == 0:
                return np.zeros(len(goalkeepers), dtype=int)
            
            team_0_centroid = team_0_players.mean(axis=0)
            team_1_centroid = team_1_players.mean(axis=0)
            
            goalkeeper_teams = []
            for gk_xy in goalkeepers_xy:
                dist_0 = np.linalg.norm(gk_xy - team_0_centroid)
                dist_1 = np.linalg.norm(gk_xy - team_1_centroid)
                goalkeeper_teams.append(0 if dist_0 < dist_1 else 1)
            
            return np.array(goalkeeper_teams, dtype=int)
            
        except Exception as e:
            print(f"Error in goalkeeper team resolution: {e}")
            return np.zeros(len(goalkeepers), dtype=int)