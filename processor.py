"""
Football AI V2 - Core Processing Logic
Simplified processor using supervision, roboflow, and ultralytics
"""

import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
import os

from models import FootballModels
from visualizer import FootballVisualizer


class FootballProcessor:
    """Main processor for football video analysis."""
    
    def __init__(self, config, device='cuda'):
        """Initialize processor."""
        self.config = config
        self.device = device
        
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
        
        print("Football AI V2 initialized successfully!")
    
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
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_width = width + 800  # Space for tactical view
        out_height = max(height, 600)
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
        
        # Train team classifier on first few frames
        self._train_team_classifier(cap, width, height)
        
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
                
                # Resize to output dimensions
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
    
    def _train_team_classifier(self, cap, width, height):
        """Train team classifier on sample frames using V1 approach."""
        print("Training team classifier...")
        
        crops = []
        stride = self.config['video'].get('stride', 30)  # Use config stride for sampling
        sample_frames = 50  # Sample more frames for better training
        
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
            
            # Detect players only (not goalkeepers for cleaner training)
            detections = self.models.detect_objects(frame)
            player_detections = detections['players']
            
            # Extract crops using supervision like in V1
            if len(player_detections) > 0:
                try:
                    import supervision as sv
                    player_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]
                    
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
        """Process a single frame."""
        # Step 1: Object detection
        detections = self.models.detect_objects(frame)
        
        # Step 2: Field detection and transformation
        field_keypoints = self.models.detect_field(frame)
        self.transformer = self.models.create_transformer(field_keypoints)
        
        # Step 3: Track objects
        human_detections = self._merge_human_detections(detections)
        if len(human_detections) > 0:
            human_detections = self.tracker.update_with_detections(human_detections)
            human_detections = self.smoother.update_with_detections(human_detections)
            
            
            # Split back into categories
            detections = self._split_human_detections(human_detections, detections)
        
        # Step 4: Team classification for players and goalkeepers  
        if len(detections['players']) > 0:
            team_ids = self.models.classify_teams(frame, detections['players'])
            detections['players'].class_id = np.array(team_ids)
        
        # Assign goalkeeper teams based on proximity to player teams (like V1)
        if len(detections['goalkeepers']) > 0 and len(detections['players']) > 0:
            gk_team_ids = self._resolve_goalkeeper_teams(detections['players'], detections['goalkeepers'])
            detections['goalkeepers'].class_id = np.array(gk_team_ids)
        elif len(detections['goalkeepers']) > 0:
            # Fallback: classify goalkeepers directly if no players
            gk_team_ids = self.models.classify_teams(frame, detections['goalkeepers'])
            detections['goalkeepers'].class_id = np.array(gk_team_ids)
        
        # Referees get a special class_id (2) for coloring
        if len(detections['referees']) > 0:
            detections['referees'].class_id = np.full(len(detections['referees']), 2, dtype=int)
        
        # Step 5: Pose estimation
        poses = {}
        if len(human_detections) > 0:
            all_poses = self.models.estimate_poses(frame, human_detections)
            # Split poses back to match our detection categories
            poses = self._split_poses_by_category(all_poses, detections)
        
        # Step 6: Segmentation  
        segments = {}
        if len(human_detections) > 0:
            all_segments = self.models.segment_players(frame, human_detections)
            # Split segments back to match our detection categories
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
        
        # Split based on original class IDs from Roboflow model
        # Class 0 = Ball, 1 = Goalkeeper, 2 = Player, 3 = Referee
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
        
        # Get ball position
        ball_pos = detections['ball'].get_anchors_coordinates(sv.Position.CENTER)
        if len(ball_pos) == 0:
            return None
        
        ball_pos = ball_pos[0]
        
        # Check all human detections
        min_distance = float('inf')
        closest_player = None
        
        for category in ['players', 'goalkeepers', 'referees']:
            if len(detections[category]) == 0:
                continue
            
            positions = detections[category].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            
            for i, pos in enumerate(positions):
                distance = np.linalg.norm(pos - ball_pos)
                if distance < min_distance and distance < 100:  # 100 pixel threshold
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
        
        # Get ball position
        ball_pos = ball_detections.get_anchors_coordinates(sv.Position.CENTER)
        if len(ball_pos) > 0:
            # Transform to pitch coordinates
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
        """Assign goalkeepers to teams based on proximity to player centroids (like V1)."""
        if len(goalkeepers) == 0 or len(players) == 0:
            return np.zeros(len(goalkeepers), dtype=int)
        
        try:
            # Get positions
            goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            
            # Get team centroids
            if not hasattr(players, 'class_id') or players.class_id is None:
                return np.zeros(len(goalkeepers), dtype=int)
            
            team_0_players = players_xy[players.class_id == 0]
            team_1_players = players_xy[players.class_id == 1]
            
            if len(team_0_players) == 0 or len(team_1_players) == 0:
                return np.zeros(len(goalkeepers), dtype=int)
            
            team_0_centroid = team_0_players.mean(axis=0)
            team_1_centroid = team_1_players.mean(axis=0)
            
            # Assign goalkeepers based on distance to team centroids
            goalkeeper_teams = []
            for gk_xy in goalkeepers_xy:
                dist_0 = np.linalg.norm(gk_xy - team_0_centroid)
                dist_1 = np.linalg.norm(gk_xy - team_1_centroid)
                goalkeeper_teams.append(0 if dist_0 < dist_1 else 1)
            
            return np.array(goalkeeper_teams, dtype=int)
            
        except Exception as e:
            print(f"Error in goalkeeper team resolution: {e}")
            return np.zeros(len(goalkeepers), dtype=int)