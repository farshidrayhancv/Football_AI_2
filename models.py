"""
Football AI V2 - Model Management
Handles Roboflow, YOLO, and SAM models
"""

import numpy as np
import supervision as sv
from inference import get_model
from ultralytics import YOLO, SAM
from transformers import AutoProcessor, SiglipVisionModel
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration
import torch
import warnings

# Filter out specific scikit-learn warnings like in V1
warnings.filterwarnings("ignore", message=".*force_all_finite.*")


class FootballModels:
    """Manages all AI models for football analysis."""
    
    def __init__(self, config, device='cuda'):
        """Initialize all models."""
        self.config = config
        self.device = device
        
        # Load models
        self._load_detection_models()
        self._load_pose_model()
        self._load_segmentation_model()
        self._load_team_classifier()
        
        # Soccer pitch configuration
        self.pitch_config = SoccerPitchConfiguration()
        
        print("All models loaded successfully!")
    
    def _load_detection_models(self):
        """Load Roboflow detection models."""
        # Player detection model
        self.player_model = get_model(
            model_id=self.config['models']['player_detection_model_id'],
            api_key=self.config['api_keys']['roboflow_api_key']
        )
        
        # Field detection model
        self.field_model = get_model(
            model_id=self.config['models']['field_detection_model_id'],
            api_key=self.config['api_keys']['roboflow_api_key']
        )
        
        print("✓ Roboflow models loaded")
    
    def _load_pose_model(self):
        """Load YOLO pose estimation model."""
        try:
            pose_model_path = self.config['models'].get('pose_model', 'yolo11n-pose.pt')
            self.pose_model = YOLO(pose_model_path)
            if self.device == 'cuda':
                self.pose_model.to(self.device)
            print(f"✓ Pose model loaded: {pose_model_path}")
        except Exception as e:
            print(f"Warning: Could not load pose model: {e}")
            self.pose_model = None
    
    def _load_segmentation_model(self):
        """Load SAM segmentation model."""
        try:
            sam_model_path = self.config['models'].get('sam_model', 'sam2.1_s.pt')
            self.sam_model = SAM(sam_model_path)
            if self.device == 'cuda':
                self.sam_model.to(self.device)
            print(f"✓ SAM model loaded: {sam_model_path}")
        except Exception as e:
            print(f"Warning: Could not load SAM model: {e}")
            self.sam_model = None
    
    def _load_team_classifier(self):
        """Load team classification model."""
        try:
            model_path = self.config['models'].get('siglip_model_path', 'google/siglip-base-patch16-224')
            hf_token = self.config['api_keys'].get('huggingface_token')
            
            self.siglip_processor = AutoProcessor.from_pretrained(model_path, token=hf_token)
            self.siglip_model = SiglipVisionModel.from_pretrained(model_path, token=hf_token)
            
            if self.device == 'cuda':
                self.siglip_model.to(self.device)
            
            # Import the proper team classifier from sports library
            from sports.common.team import TeamClassifier
            self.team_classifier = None  # Will be trained during video processing
            
            print(f"✓ Team classifier model loaded: {model_path}")
        except Exception as e:
            print(f"Warning: Could not load team classifier: {e}")
            self.siglip_model = None
            self.siglip_processor = None
            self.team_classifier = None
    
    def detect_objects(self, frame):
        """Detect all objects in frame using Roboflow model."""
        try:
            # Run inference
            result = self.player_model.infer(
                frame,
                confidence=self.config['detection']['confidence_threshold']
            )[0]
            
            # Convert to supervision format
            detections = sv.Detections.from_inference(result)
            
            # Separate by class (based on your V1 configuration)
            ball_detections = detections[detections.class_id == 0]  # Ball
            goalkeeper_detections = detections[detections.class_id == 1]  # Goalkeeper
            player_detections = detections[detections.class_id == 2]  # Player
            referee_detections = detections[detections.class_id == 3]  # Referee
            
            # Apply padding to ball detections
            if len(ball_detections) > 0:
                ball_detections.xyxy = sv.pad_boxes(ball_detections.xyxy, px=10)
            
            return {
                'players': player_detections,
                'goalkeepers': goalkeeper_detections,
                'referees': referee_detections,
                'ball': ball_detections
            }
            
        except Exception as e:
            print(f"Error in object detection: {e}")
            return {
                'players': sv.Detections.empty(),
                'goalkeepers': sv.Detections.empty(),
                'referees': sv.Detections.empty(),
                'ball': sv.Detections.empty()
            }
    
    def detect_field(self, frame):
        """Detect field keypoints using Roboflow model."""
        try:
            result = self.field_model.infer(
                frame,
                confidence=self.config['detection']['keypoint_confidence_threshold']
            )[0]
            
            return sv.KeyPoints.from_inference(result)
            
        except Exception as e:
            print(f"Error in field detection: {e}")
            return None
    
    def create_transformer(self, keypoints):
        """Create coordinate transformer from field keypoints."""
        if keypoints is None or len(keypoints.xy) == 0:
            return None
        
        try:
            # Filter keypoints by confidence
            confidence_threshold = self.config['detection']['keypoint_confidence_threshold']
            filter_mask = keypoints.confidence[0] > confidence_threshold
            frame_points = keypoints.xy[0][filter_mask]
            
            # Get corresponding pitch points
            pitch_points = np.array(self.pitch_config.vertices)[filter_mask]
            
            if len(frame_points) >= 4:
                transformer = ViewTransformer(
                    source=frame_points,
                    target=pitch_points
                )
                return transformer
            
        except Exception as e:
            print(f"Error creating transformer: {e}")
        
        return None
    
    def estimate_poses(self, frame, detections):
        """Estimate poses for detected humans."""
        if self.pose_model is None or len(detections) == 0:
            return []
        
        poses = []
        
        for box in detections.xyxy:
            try:
                # Extract padded crop
                x1, y1, x2, y2 = map(int, box)
                
                # Add padding (10%)
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                
                x1_pad = max(0, x1 - pad_x)
                y1_pad = max(0, y1 - pad_y)
                x2_pad = min(frame.shape[1], x2 + pad_x)
                y2_pad = min(frame.shape[0], y2 + pad_y)
                
                crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if crop.size == 0:
                    poses.append(None)
                    continue
                
                # Run pose estimation
                results = self.pose_model(crop, verbose=False)
                
                if (results and len(results) > 0 and 
                    hasattr(results[0], 'keypoints') and 
                    results[0].keypoints is not None and
                    len(results[0].keypoints.data) > 0):
                    
                    # Get keypoints
                    kpts = results[0].keypoints.data[0].cpu().numpy()
                    
                    # Adjust coordinates back to full frame
                    pose_data = {
                        'keypoints': kpts[:, :2].copy(),
                        'confidence': kpts[:, 2].copy()
                    }
                    
                    pose_data['keypoints'][:, 0] += x1_pad
                    pose_data['keypoints'][:, 1] += y1_pad
                    
                    poses.append(pose_data)
                else:
                    poses.append(None)
                    
            except Exception as e:
                print(f"Error in pose estimation: {e}")
                poses.append(None)
        
        return poses
    
    def segment_players(self, frame, detections):
        """Segment players using SAM."""
        if self.sam_model is None or len(detections) == 0:
            return []
        
        segments = []
        
        for box in detections.xyxy:
            try:
                # Run SAM with box prompt
                results = self.sam_model(
                    frame,
                    bboxes=[box.tolist()],
                    verbose=False
                )
                
                if (results and len(results) > 0 and
                    hasattr(results[0], 'masks') and
                    results[0].masks is not None and
                    len(results[0].masks.data) > 0):
                    
                    mask = results[0].masks.data[0].cpu().numpy()
                    segments.append(mask)
                else:
                    segments.append(None)
                    
            except Exception as e:
                print(f"Error in segmentation: {e}")
                segments.append(None)
        
        return segments
    
    def train_team_classifier(self, player_crops):
        """Train team classifier on player crops using sports library."""
        if self.siglip_model is None or not player_crops:
            print("Warning: Cannot train team classifier - missing model or no crops")
            return
        
        try:
            # Import the proper team classifier from sports library
            from sports.common.team import TeamClassifier
            
            # Initialize the team classifier with proper device
            self.team_classifier = TeamClassifier(device=self.device)
            
            # Train the classifier with player crops
            self.team_classifier.fit(player_crops)
            
            print(f"✓ Team classifier trained on {len(player_crops)} player crops using sports library")
            
        except Exception as e:
            print(f"Error training team classifier: {e}")
            self.team_classifier = None
    
    def classify_teams(self, frame, player_detections):
        """Classify players into teams using stable sports library classifier."""
        if self.team_classifier is None or len(player_detections) == 0:
            print("Warning: Team classifier not trained or no player detections")
            return np.zeros(len(player_detections), dtype=int)
        
        try:
            # Extract crops using supervision's crop_image function like in V1
            import supervision as sv
            player_crops = []
            
            for xyxy in player_detections.xyxy:
                try:
                    crop = sv.crop_image(frame, xyxy)
                    if crop.size > 0:
                        player_crops.append(crop)
                    else:
                        # Fallback for invalid crops
                        player_crops.append(np.zeros((32, 32, 3), dtype=np.uint8))
                except Exception as e:
                    print(f"Error cropping player: {e}")
                    player_crops.append(np.zeros((32, 32, 3), dtype=np.uint8))
            
            if not player_crops:
                return np.zeros(len(player_detections), dtype=int)
            
            # Use the sports library's stable team classifier
            team_ids = self.team_classifier.predict(player_crops)
            
            return np.array(team_ids, dtype=int)
            
        except Exception as e:
            print(f"Error in team classification: {e}")
            return np.zeros(len(player_detections), dtype=int)