"""
Football AI V2 - Model Management
Handles Roboflow, YOLO, and SAM models
Now with Hybrid Pose Estimation: Full Frame + SAHI + Dummy Poses
"""

import numpy as np
import supervision as sv

# Optional import for Roboflow - only needed if using API models
try:
    from inference import get_model
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    print("Note: Roboflow inference not installed. Using custom models only.")

from ultralytics import YOLO, SAM
from transformers import AutoProcessor, SiglipVisionModel
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration
import torch
import warnings
import cv2

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
        self._load_field_detection_model()
        
        # Soccer pitch configuration
        self.pitch_config = SoccerPitchConfiguration()
        
        # Pose estimation configuration from config
        pose_config = self.config.get('pose_estimation', {})
        self.pose_confidence_threshold = pose_config.get('confidence_threshold', 0.2)
        
        # Pose smoothing configuration
        smoothing_config = pose_config.get('smoothing', {})
        self.enable_pose_smoothing = smoothing_config.get('enable', True)
        self.smoothing_window = smoothing_config.get('window_size', 5)
        
        # Initialize pose smoothing storage
        if self.enable_pose_smoothing:
            self.pose_history = {}  # {tracker_id: [pose1, pose2, ...]}
            self.max_history_length = self.smoothing_window
        
        if self.config.get('debug', {}).get('verbose_pose_estimation', False):
            print(f"✓ Pose confidence threshold: {self.pose_confidence_threshold}")
            if self.enable_pose_smoothing:
                print(f"✓ Pose smoothing enabled with window size: {self.smoothing_window}")
        
        print("All models loaded successfully!")
    
    def _load_detection_models(self):
        """Load object detection models - either Roboflow or custom YOLO."""
        try:
            # Check if we should use custom YOLO model for object detection
            use_custom_detection = self.config['models'].get('use_custom_object_detection', False)
            
            if use_custom_detection:
                # Load custom YOLO11n object detection model
                detection_model_path = self.config['models'].get('custom_object_model', 'yolo11n-football.pt')
                self.player_model = YOLO(detection_model_path)
                if self.device == 'cuda':
                    self.player_model.to(self.device)
                print(f"✓ Custom YOLO object detection model loaded: {detection_model_path}")
                self.use_custom_object_model = True
                
                # Get class mapping from model
                self.class_names = self.player_model.names if hasattr(self.player_model, 'names') else {}
                print(f"  Classes: {self.class_names}")
                
                # Define class ID mapping (adjust based on your model's training)
                # Default mapping - update in config if different
                self.class_mapping = self.config['models'].get('object_class_mapping', {
                    'ball': 0,
                    'goalkeeper': 1, 
                    'player': 2,
                    'referee': 3
                })
                
            else:
                # Original Roboflow model
                if not ROBOFLOW_AVAILABLE:
                    raise ImportError("Roboflow inference not installed. Please install with: pip install inference-gpu")
                
                self.player_model = get_model(
                    model_id=self.config['models']['player_detection_model_id'],
                    api_key=self.config['api_keys']['roboflow_api_key']
                )
                print("✓ Roboflow player detection model loaded")
                self.use_custom_object_model = False
                
                # Roboflow class mapping (standard)
                self.class_mapping = {
                    'ball': 0,
                    'goalkeeper': 1,
                    'player': 2,
                    'referee': 3
                }
                
        except Exception as e:
            print(f"Warning: Could not load object detection model: {e}")
            
            if use_custom_detection:
                print("Falling back to Roboflow model...")
                try:
                    if not ROBOFLOW_AVAILABLE:
                        raise ImportError("Roboflow inference not installed")
                    
                    self.player_model = get_model(
                        model_id=self.config['models']['player_detection_model_id'],
                        api_key=self.config['api_keys']['roboflow_api_key']
                    )
                    self.use_custom_object_model = False
                    self.class_mapping = {
                        'ball': 0,
                        'goalkeeper': 1,
                        'player': 2,
                        'referee': 3
                    }
                except:
                    print("Error: Neither custom model nor Roboflow available for object detection")
                    self.player_model = None
                    self.use_custom_object_model = False
            else:
                self.player_model = None
                self.use_custom_object_model = False
    
    def _load_field_detection_model(self):
        """Load field detection model - either Roboflow or custom YOLO."""
        try:
            # Check if we should use custom YOLO model
            use_custom_yolo = self.config['models'].get('use_custom_field_detection', False)
            
            if use_custom_yolo:
                # Load custom YOLO11n-pose model trained on pitch keypoints
                field_model_path = self.config['models'].get('custom_field_model', 'yolo11n-pose-pitch.pt')
                self.field_model = YOLO(field_model_path)
                if self.device == 'cuda':
                    self.field_model.to(self.device)
                print(f"✓ Custom YOLO field detection model loaded: {field_model_path}")
                self.use_custom_field_model = True
                print(f"  Model loaded successfully")
                
            else:
                # Original Roboflow model
                if not ROBOFLOW_AVAILABLE:
                    raise ImportError("Roboflow inference not installed. Please install with: pip install inference-gpu")
                
                self.field_model = get_model(
                    model_id=self.config['models']['field_detection_model_id'],
                    api_key=self.config['api_keys']['roboflow_api_key']
                )
                print("✓ Roboflow field detection model loaded")
                self.use_custom_field_model = False
                
        except Exception as e:
            print(f"Warning: Could not load field detection model: {e}")
            
            if use_custom_yolo:
                print("Falling back to Roboflow model...")
                # Fallback to Roboflow
                try:
                    if not ROBOFLOW_AVAILABLE:
                        raise ImportError("Roboflow inference not installed")
                    
                    self.field_model = get_model(
                        model_id=self.config['models']['field_detection_model_id'],
                        api_key=self.config['api_keys']['roboflow_api_key']
                    )
                    self.use_custom_field_model = False
                except:
                    print("Error: Neither custom model nor Roboflow available for field detection")
                    self.field_model = None
                    self.use_custom_field_model = False
            else:
                self.field_model = None
                self.use_custom_field_model = False
    
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
    
    def calculate_adaptive_padding(self, box_width, box_height, mode='pose'):
        """
        Calculate adaptive padding based on bounding box size.
        Fully configurable through config.yaml settings.
        
        Args:
            box_width: Width of the bounding box
            box_height: Height of the bounding box  
            mode: 'pose' or 'segmentation' for different padding strategies
        
        Returns:
            pad_x, pad_y: Padding in pixels for x and y dimensions
        """
        # Get adaptive padding config section
        adaptive_config = self.config['detection'].get('adaptive_padding', {})
        
        # Check if adaptive padding is enabled
        if not adaptive_config.get('enable', False):
            # Use fixed padding from config or default 10%
            fixed_ratio = adaptive_config.get('fixed_padding_ratio', 0.1)
            return int(box_width * fixed_ratio), int(box_height * fixed_ratio)
        
        # Get frame dimensions from config or use defaults
        resolution_config = self.config.get('resolution', {})
        if 'original' in resolution_config:
            frame_width, frame_height = resolution_config['original']
        else:
            # Use the video's original resolution if specified
            frame_width = adaptive_config.get('reference_frame_width', 1920)
            frame_height = adaptive_config.get('reference_frame_height', 1080)
        
        # Calculate normalized box size
        box_area = box_width * box_height
        frame_area = frame_width * frame_height
        normalized_size = min(box_area / frame_area, 1.0)
        
        # Get mode-specific padding configuration
        if mode == 'pose':
            mode_config = adaptive_config.get('pose_padding', {})
            # Default padding ratios if not specified in config
            default_ratios = {
                'very_small': 0.5,
                'small': 0.3,
                'medium': 0.2,
                'large': 0.1
            }
        else:  # segmentation
            mode_config = adaptive_config.get('segmentation_padding', {})
            default_ratios = {
                'very_small': 0.3,
                'small': 0.2,
                'medium': 0.15,
                'large': 0.1
            }
        
        # Get size thresholds from config
        thresholds = adaptive_config.get('size_thresholds', {})
        very_small_threshold = thresholds.get('very_small', 0.001)
        small_threshold = thresholds.get('small', 0.005)
        medium_threshold = thresholds.get('medium', 0.02)
        
        # Determine size category and get corresponding padding ratio
        if normalized_size < very_small_threshold:
            size_category = 'very_small'
            padding_ratio = mode_config.get('very_small', default_ratios['very_small'])
        elif normalized_size < small_threshold:
            size_category = 'small'
            padding_ratio = mode_config.get('small', default_ratios['small'])
        elif normalized_size < medium_threshold:
            size_category = 'medium'
            padding_ratio = mode_config.get('medium', default_ratios['medium'])
        else:
            size_category = 'large'
            padding_ratio = mode_config.get('large', default_ratios['large'])
        
        # Apply distance-based scaling if enabled
        if adaptive_config.get('use_distance_estimation', False):
            distance_scaling = adaptive_config.get('distance_based_scaling', {})
            scale_factor = distance_scaling.get(size_category.replace('_small', '').replace('very_', 'very_'), 1.0)
            padding_ratio *= scale_factor
        
        # Get minimum padding pixels from config
        min_pad_x = mode_config.get('min_pixels_x', 20 if mode == 'pose' else 15)
        min_pad_y = mode_config.get('min_pixels_y', 30 if mode == 'pose' else 20)
        
        # Calculate padding with minimums
        pad_x = max(int(box_width * padding_ratio), min_pad_x)
        pad_y = max(int(box_height * padding_ratio), min_pad_y)
        
        # Apply global multiplier if specified
        multiplier = adaptive_config.get('adaptive_padding_multiplier', 1.0)
        pad_x = int(pad_x * multiplier)
        pad_y = int(pad_y * multiplier)
        
        # Apply maximum padding limits if specified
        if 'max_padding_ratio' in adaptive_config:
            max_ratio = adaptive_config['max_padding_ratio']
            pad_x = min(pad_x, int(box_width * max_ratio))
            pad_y = min(pad_y, int(box_height * max_ratio))
        
        if 'max_pixels_x' in mode_config:
            pad_x = min(pad_x, mode_config['max_pixels_x'])
        if 'max_pixels_y' in mode_config:
            pad_y = min(pad_y, mode_config['max_pixels_y'])
        
        # Debug logging if enabled
        if self.config.get('debug', {}).get('verbose_padding', False):
            print(f"Adaptive padding: size={box_width}x{box_height}, "
                  f"normalized={normalized_size:.4f}, category={size_category}, "
                  f"ratio={padding_ratio:.2f}, padding={pad_x}x{pad_y}px")
        
        return pad_x, pad_y
    
    def detect_objects(self, frame):
        """Detect all objects in frame using either Roboflow or custom YOLO model."""
        if self.player_model is None:
            print("Warning: No object detection model loaded")
            return {
                'players': sv.Detections.empty(),
                'goalkeepers': sv.Detections.empty(),
                'referees': sv.Detections.empty(),
                'ball': sv.Detections.empty()
            }
        
        try:
            if self.use_custom_object_model:
                # Use custom YOLO model
                results = self.player_model(frame, 
                                          conf=self.config['detection']['confidence_threshold'],
                                          verbose=False)
                
                if results and len(results) > 0:
                    # Convert YOLO results to supervision format
                    detections = sv.Detections.from_ultralytics(results[0])
                    
                    # Separate by class using configured mapping
                    ball_mask = detections.class_id == self.class_mapping.get('ball', 0)
                    goalkeeper_mask = detections.class_id == self.class_mapping.get('goalkeeper', 1)
                    player_mask = detections.class_id == self.class_mapping.get('player', 2)
                    referee_mask = detections.class_id == self.class_mapping.get('referee', 3)
                    
                    ball_detections = detections[ball_mask]
                    goalkeeper_detections = detections[goalkeeper_mask]
                    player_detections = detections[player_mask]
                    referee_detections = detections[referee_mask]
                    
                    # Apply padding to ball detections
                    if len(ball_detections) > 0:
                        ball_detections.xyxy = sv.pad_boxes(ball_detections.xyxy, px=10)
                    
                    return {
                        'players': player_detections,
                        'goalkeepers': goalkeeper_detections,
                        'referees': referee_detections,
                        'ball': ball_detections
                    }
                else:
                    return {
                        'players': sv.Detections.empty(),
                        'goalkeepers': sv.Detections.empty(),
                        'referees': sv.Detections.empty(),
                        'ball': sv.Detections.empty()
                    }
                    
            else:
                # Original Roboflow implementation
                if not ROBOFLOW_AVAILABLE:
                    print("Error: Roboflow inference not available but API model selected")
                    return {
                        'players': sv.Detections.empty(),
                        'goalkeepers': sv.Detections.empty(),
                        'referees': sv.Detections.empty(),
                        'ball': sv.Detections.empty()
                    }
                
                result = self.player_model.infer(
                    frame,
                    confidence=self.config['detection']['confidence_threshold']
                )[0]
                
                # Convert to supervision format
                detections = sv.Detections.from_inference(result)
                
                # Separate by class (Roboflow standard mapping)
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
        """Detect field keypoints using either Roboflow or custom YOLO model."""
        if self.field_model is None:
            return None
        
        verbose = self.config.get('debug', {}).get('verbose_field_detection', False)
        
        try:
            if self.use_custom_field_model:
                # Use custom YOLO11n-pose model
                # Get the resolution for field detection
                field_res = self.config.get('resolution', {}).get('field_detection', None)
                
                # Run inference
                results = self.field_model(frame, verbose=False)
                
                if (results and len(results) > 0 and 
                    hasattr(results[0], 'keypoints') and 
                    results[0].keypoints is not None and
                    hasattr(results[0].keypoints, 'data') and
                    len(results[0].keypoints.data) > 0):
                    
                    # Extract keypoints from YOLO pose format
                    kpts_data = results[0].keypoints.data[0].cpu().numpy()  # Shape: (num_keypoints, 3)
                    
                    # Convert to supervision KeyPoints format
                    # YOLO gives us (x, y, confidence) for each keypoint
                    xy_points = kpts_data[:, :2]  # Get x, y coordinates
                    confidences = kpts_data[:, 2]  # Get confidence scores
                    
                    # Create supervision KeyPoints object
                    # Note: supervision expects shape (num_objects, num_keypoints, 2) for xy
                    # and (num_objects, num_keypoints) for confidence
                    keypoints = sv.KeyPoints(
                        xy=xy_points[np.newaxis, ...],  # Add batch dimension
                        confidence=confidences[np.newaxis, ...]  # Add batch dimension
                    )
                    
                    if verbose:
                        num_detected = len(xy_points)
                        num_confident = np.sum(confidences > self.config['detection']['keypoint_confidence_threshold'])
                        print(f"Custom YOLO: Detected {num_detected} keypoints, {num_confident} high-confidence")
                    
                    return keypoints
                else:
                    if verbose:
                        print("No keypoints detected in frame")
                    return None
                    
            else:
                # Original Roboflow implementation
                if not ROBOFLOW_AVAILABLE:
                    print("Error: Roboflow inference not available but API model selected")
                    return None
                
                result = self.field_model.infer(
                    frame,
                    confidence=self.config['detection']['keypoint_confidence_threshold']
                )[0]
                keypoints = sv.KeyPoints.from_inference(result)
                
                if verbose and keypoints is not None:
                    num_detected = len(keypoints.xy[0]) if len(keypoints.xy) > 0 else 0
                    if num_detected > 0:
                        num_confident = np.sum(keypoints.confidence[0] > self.config['detection']['keypoint_confidence_threshold'])
                        print(f"Roboflow: Detected {num_detected} keypoints, {num_confident} high-confidence")
                
                return keypoints
                
        except Exception as e:
            print(f"Error in field detection: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def create_transformer(self, keypoints):
        """Create coordinate transformer from field keypoints."""
        if keypoints is None or len(keypoints.xy) == 0:
            return None
        
        verbose = self.config.get('debug', {}).get('verbose_transformer', False)
        
        try:
            # Filter keypoints by confidence
            confidence_threshold = self.config['detection']['keypoint_confidence_threshold']
            all_keypoints = keypoints.xy[0]
            all_confidences = keypoints.confidence[0]
            
            # Get all pitch vertices
            all_pitch_vertices = np.array(self.pitch_config.vertices)
            
            if self.use_custom_field_model:
                # For custom YOLO model, use keypoint mapping if provided
                keypoint_mapping = self.config.get('keypoint_mapping', {}).get('mapping', {})
                
                if keypoint_mapping:
                    # Use explicit mapping from config
                    frame_points = []
                    pitch_points = []
                    
                    for model_idx, pitch_idx in keypoint_mapping.items():
                        model_idx = int(model_idx)
                        pitch_idx = int(pitch_idx)
                        
                        # Check if this keypoint has high enough confidence
                        if (model_idx < len(all_keypoints) and 
                            all_confidences[model_idx] > confidence_threshold):
                            
                            frame_points.append(all_keypoints[model_idx])
                            pitch_points.append(all_pitch_vertices[pitch_idx])
                    
                    frame_points = np.array(frame_points)
                    pitch_points = np.array(pitch_points)
                    
                    # Only print in debug mode or if there's an issue
                    if len(frame_points) < 4:
                        print(f"Warning: Only {len(frame_points)} keypoints with mapping")
                    elif verbose:
                        print(f"Using mapped keypoints: {len(frame_points)} high-confidence points")
                    
                else:
                    # No explicit mapping - assume keypoints match pitch vertices order
                    filter_mask = all_confidences > confidence_threshold
                    frame_points = all_keypoints[filter_mask]
                    
                    # Only use pitch vertices that correspond to detected keypoints
                    valid_indices = np.where(filter_mask)[0]
                    if len(valid_indices) <= len(all_pitch_vertices):
                        pitch_points = all_pitch_vertices[valid_indices]
                    else:
                        # More keypoints than pitch vertices - truncate
                        pitch_points = all_pitch_vertices[:len(frame_points)]
                    
                    # Only print in debug mode or if there's an issue
                    if len(frame_points) < 4:
                        print(f"Warning: Only {len(frame_points)} keypoints without mapping")
                    elif verbose:
                        print(f"Using direct mapping: {len(frame_points)} high-confidence points")
                    
            else:
                # For Roboflow model, use standard filtering
                filter_mask = all_confidences > confidence_threshold
                frame_points = all_keypoints[filter_mask]
                pitch_points = all_pitch_vertices[filter_mask]
            
            # Validate we have enough points
            if len(frame_points) < 4:
                print(f"Not enough high-confidence keypoints: {len(frame_points)}")
                return None
            
            if len(frame_points) != len(pitch_points):
                print(f"Warning: Mismatch between frame points ({len(frame_points)}) and pitch points ({len(pitch_points)})")
                # Use minimum common length
                min_len = min(len(frame_points), len(pitch_points))
                frame_points = frame_points[:min_len]
                pitch_points = pitch_points[:min_len]
            
            # Create transformer
            transformer = ViewTransformer(
                source=frame_points,
                target=pitch_points
            )
            
            # Validate transformer by testing a known point
            test_point = np.array([[frame_points[0][0], frame_points[0][1]]])
            try:
                transformed = transformer.transform_points(test_point)
                # Success - transformer is working
                if verbose:
                    print(f"✓ Transformer created successfully with {len(frame_points)} points")
            except Exception as e:
                print(f"Warning: Transformer validation failed: {e}")
            
            return transformer
                
        except Exception as e:
            print(f"Error creating transformer: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            return None
    
    def _generate_dummy_pose(self, box):
        """Generate a dummy standing pose for a bounding box."""
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        bottom_y = y2
        top_y = y1
        height = y2 - y1
        width = x2 - x1
        
        # Create COCO-style standing pose (17 keypoints)
        head_y = top_y + height * 0.1
        shoulder_y = top_y + height * 0.25
        elbow_y = shoulder_y + height * 0.2
        wrist_y = elbow_y + height * 0.15
        hip_y = top_y + height * 0.55
        knee_y = hip_y + height * 0.25
        ankle_y = bottom_y - height * 0.05
        
        keypoints = np.array([
            [center_x, head_y],  # 0: nose
            [center_x - width*0.05, head_y - height*0.02],  # 1: left_eye
            [center_x + width*0.05, head_y - height*0.02],  # 2: right_eye
            [center_x - width*0.08, head_y],  # 3: left_ear
            [center_x + width*0.08, head_y],  # 4: right_ear
            [center_x - width*0.15, shoulder_y],  # 5: left_shoulder
            [center_x + width*0.15, shoulder_y],  # 6: right_shoulder
            [center_x - width*0.12, elbow_y],  # 7: left_elbow
            [center_x + width*0.12, elbow_y],  # 8: right_elbow
            [center_x - width*0.08, wrist_y],  # 9: left_wrist
            [center_x + width*0.08, wrist_y],  # 10: right_wrist
            [center_x - width*0.1, hip_y],  # 11: left_hip
            [center_x + width*0.1, hip_y],  # 12: right_hip
            [center_x - width*0.08, knee_y],  # 13: left_knee
            [center_x + width*0.08, knee_y],  # 14: right_knee
            [center_x - width*0.06, ankle_y],  # 15: left_ankle
            [center_x + width*0.06, ankle_y],  # 16: right_ankle
        ])
        
        confidence = np.full(17, 0.6)
        
        return {
            'keypoints': keypoints,
            'confidence': confidence,
            'is_dummy': True
        }
    
    def _run_sahi_pose_estimation(self, frame, existing_poses, detections):
        """Run SAHI pose estimation using supervision's InferenceSlicer."""
        if self.pose_model is None:
            return existing_poses
        
        try:
            print(f"🔍 SAHI: Starting SAHI pose estimation on {frame.shape[1]}x{frame.shape[0]} frame")
            
            # Create SAHI slicer with proper 2x2 grid (ensure slices are actually smaller than frame)
            slice_width = max(320, frame.shape[1] // 2)  # Minimum 320px slices
            slice_height = max(320, frame.shape[0] // 2)  # Minimum 320px slices
            
            # Only do SAHI if frame is large enough to benefit from slicing
            if frame.shape[1] <= slice_width * 1.5 or frame.shape[0] <= slice_height * 1.5:
                print(f"🔍 SAHI: Frame too small for effective slicing, skipping SAHI")
                return existing_poses
            
            print(f"🔍 SAHI: Using slice size {slice_width}x{slice_height}")
            
            slicer = sv.InferenceSlicer(
                callback=self._pose_callback,
                slice_wh=(slice_width, slice_height),
                overlap_wh=(64, 64),  # Fixed overlap in pixels instead of ratio
                iou_threshold=0.5
            )
            
            # Run SAHI pose estimation
            sahi_results = slicer(frame)
            print(f"🔍 SAHI: Slicer returned {len(sahi_results)} detections")
            
            # The slicer returns sv.Detections, but we need to run pose estimation
            # on the detected regions to get keypoints
            sahi_poses = []
            
            if len(sahi_results) > 0:
                print(f"🔍 SAHI: Processing {len(sahi_results)} SAHI detections for pose estimation")
                
                # For each detection from SAHI, extract the region and run pose estimation
                for i, bbox in enumerate(sahi_results.xyxy):
                    x1, y1, x2, y2 = bbox.astype(int)
                    
                    # Add padding around the detection
                    pad = 20
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(frame.shape[1], x2 + pad)
                    y2 = min(frame.shape[0], y2 + pad)
                    
                    # Extract region
                    region = frame[y1:y2, x1:x2]
                    
                    if region.size == 0:
                        continue
                    
                    # Run pose estimation on this region
                    try:
                        results = self.pose_model(region, verbose=False)
                        
                        if (results and len(results) > 0 and 
                            hasattr(results[0], 'keypoints') and 
                            results[0].keypoints is not None and
                            len(results[0].keypoints.data) > 0):
                            
                            poses_data = results[0].keypoints.data.cpu().numpy()
                            
                            for pose_data in poses_data:
                                if len(pose_data) > 0:
                                    keypoints = pose_data[:, :2].copy()
                                    confidence = pose_data[:, 2].copy()
                                    
                                    # Adjust keypoints back to full frame coordinates
                                    keypoints[:, 0] += x1
                                    keypoints[:, 1] += y1
                                    
                                    # Check confidence threshold
                                    valid_confidences = confidence[confidence > 0]
                                    if len(valid_confidences) == 0:
                                        continue
                                        
                                    avg_confidence = np.mean(valid_confidences)
                                    if avg_confidence >= self.pose_confidence_threshold:
                                        sahi_poses.append({
                                            'keypoints': keypoints,
                                            'confidence': confidence,
                                            'avg_confidence': avg_confidence,
                                            'is_sahi': True
                                        })
                                        print(f"🔍 SAHI: Added pose from region {i} with confidence {avg_confidence:.2f}")
                    
                    except Exception as e:
                        print(f"🔍 SAHI: Error processing region {i}: {e}")
                        continue
            
            print(f"🔍 SAHI: Total valid SAHI poses: {len(sahi_poses)}")
            
            # Match SAHI poses with existing detections
            updated_poses = existing_poses.copy()
            matched_count = 0
            
            for i, detection_box in enumerate(detections.xyxy):
                x1, y1, x2, y2 = detection_box
                box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                
                best_sahi_pose = None
                best_distance = float('inf')
                
                # Find best matching SAHI pose
                for j, sahi_pose in enumerate(sahi_poses):
                    valid_keypoints = sahi_pose['keypoints'][sahi_pose['confidence'] > 0.3]
                    if len(valid_keypoints) == 0:
                        continue
                    
                    pose_center = np.mean(valid_keypoints, axis=0)
                    distance = np.linalg.norm(pose_center - box_center)
                    
                    # Check if pose is within detection box
                    if (x1 <= pose_center[0] <= x2 and y1 <= pose_center[1] <= y2 and 
                        distance < best_distance):
                        best_distance = distance
                        best_sahi_pose = sahi_pose
                        print(f"🔍 SAHI: Found matching pose for detection {i} at distance {distance:.1f}")
                
                # Update or assign pose
                if best_sahi_pose:
                    if (i < len(updated_poses) and updated_poses[i] is not None and 
                        not updated_poses[i].get('is_dummy', False)):
                        # Improve existing pose if SAHI has higher confidence
                        existing_confidence = np.mean(updated_poses[i]['confidence'][updated_poses[i]['confidence'] > 0])
                        if best_sahi_pose['avg_confidence'] > existing_confidence:
                            updated_poses[i] = {
                                'keypoints': best_sahi_pose['keypoints'],
                                'confidence': best_sahi_pose['confidence'],
                                'box_padding': updated_poses[i].get('box_padding', (0, 0)),
                                'box_size': updated_poses[i].get('box_size', (x2-x1, y2-y1)),
                                'improved_by_sahi': True
                            }
                            matched_count += 1
                            print(f"🔍 SAHI: Improved existing pose for detection {i} (old: {existing_confidence:.2f}, new: {best_sahi_pose['avg_confidence']:.2f})")
                    else:
                        # Assign new pose from SAHI
                        if i < len(updated_poses):
                            updated_poses[i] = {
                                'keypoints': best_sahi_pose['keypoints'],
                                'confidence': best_sahi_pose['confidence'],
                                'box_padding': (0, 0),
                                'box_size': (x2-x1, y2-y1),
                                'from_sahi': True
                            }
                            matched_count += 1
                            print(f"🔍 SAHI: Assigned new pose to detection {i} (confidence: {best_sahi_pose['avg_confidence']:.2f})")
            
            print(f"🔍 SAHI: Matched {matched_count} SAHI poses to detections")
            return updated_poses
            
        except Exception as e:
            print(f"❌ Error in SAHI pose estimation: {e}")
            import traceback
            traceback.print_exc()
            return existing_poses
    
    def estimate_poses(self, frame, detections):
        """
        Hybrid pose estimation with 2 stages:
        1. Full frame pose estimation
        2. Individual box pose estimation for each detection
        3. Dummy poses for remaining players
        """
        if self.pose_model is None or len(detections) == 0:
            if self.config.get('debug', {}).get('verbose_pose_estimation', False):
                print(f"🔍 Pose: No pose model or no detections ({len(detections)})")
            return []
        
        verbose = self.config.get('debug', {}).get('verbose_pose_estimation', False)
        pose_config = self.config.get('pose_estimation', {})
        
        if verbose:
            print(f"🔍 Pose: Starting pose estimation for {len(detections)} detections")
        
        poses = [None] * len(detections)
        
        # Stage 1: Full frame pose estimation
        full_frame_count = 0
        try:
            if verbose:
                print(f"🔍 Stage 1: Full frame pose estimation...")
            results = self.pose_model(frame, verbose=False)
            
            if (results and len(results) > 0 and 
                hasattr(results[0], 'keypoints') and 
                results[0].keypoints is not None and
                len(results[0].keypoints.data) > 0):
                
                full_frame_poses = results[0].keypoints.data.cpu().numpy()
                if verbose:
                    print(f"🔍 Stage 1: Found {len(full_frame_poses)} poses in full frame")
                
                # Match poses to detections
                for i, detection_box in enumerate(detections.xyxy):
                    x1, y1, x2, y2 = detection_box
                    box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                    
                    best_pose_idx = None
                    best_distance = float('inf')
                    
                    for pose_idx, pose_data in enumerate(full_frame_poses):
                        if len(pose_data) == 0:
                            continue
                        
                        keypoints = pose_data[:, :2]
                        confidence = pose_data[:, 2]
                        
                        valid_confidences = confidence[confidence > 0]
                        if len(valid_confidences) == 0:
                            continue
                            
                        avg_confidence = np.mean(valid_confidences)
                        if avg_confidence < self.pose_confidence_threshold:
                            continue
                        
                        valid_keypoints = keypoints[confidence > 0.3]
                        if len(valid_keypoints) == 0:
                            continue
                        
                        pose_center = np.mean(valid_keypoints, axis=0)
                        distance = np.linalg.norm(pose_center - box_center)
                        
                        if (x1 <= pose_center[0] <= x2 and y1 <= pose_center[1] <= y2 and 
                            distance < best_distance):
                            best_distance = distance
                            best_pose_idx = pose_idx
                    
                    if best_pose_idx is not None:
                        pose_data = full_frame_poses[best_pose_idx]
                        poses[i] = {
                            'keypoints': pose_data[:, :2].copy(),
                            'confidence': pose_data[:, 2].copy(),
                            'from_full_frame': True
                        }
                        full_frame_count += 1
                        if verbose:
                            print(f"🔍 Stage 1: Matched pose to detection {i}")
                        full_frame_poses[best_pose_idx] = np.array([])  # Mark as used
        
        except Exception as e:
            if verbose:
                print(f"❌ Stage 1 error: {e}")
        
        if verbose:
            print(f"🔍 Stage 1: {full_frame_count}/{len(detections)} poses from full frame")
        
        # Stage 2: Individual box pose estimation
        box_count = 0
        if verbose:
            print(f"🔍 Stage 2: Individual box pose estimation...")
        
        # Get padding settings from config
        padding_ratio = pose_config.get('box_padding_ratio', 0.3)
        min_padding = pose_config.get('min_padding_pixels', [20, 30])
        
        for i, detection_box in enumerate(detections.xyxy):
            if poses[i] is not None:  # Already has pose from full frame
                continue
            
            try:
                x1, y1, x2, y2 = map(int, detection_box)
                box_width = x2 - x1
                box_height = y2 - y1
                
                if box_width < 5 or box_height < 5:
                    continue
                
                # Add padding based on config
                pad_x = max(min_padding[0], int(box_width * padding_ratio))
                pad_y = max(min_padding[1], int(box_height * padding_ratio))
                x1_pad = max(0, x1 - pad_x)
                y1_pad = max(0, y1 - pad_y)
                x2_pad = min(frame.shape[1], x2 + pad_x)
                y2_pad = min(frame.shape[0], y2 + pad_y)
                
                # Extract crop
                crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                if crop.size == 0:
                    continue
                
                # Run pose estimation on crop
                results = self.pose_model(crop, verbose=False)
                
                if (results and len(results) > 0 and 
                    hasattr(results[0], 'keypoints') and 
                    results[0].keypoints is not None and
                    len(results[0].keypoints.data) > 0):
                    
                    crop_poses = results[0].keypoints.data.cpu().numpy()
                    
                    # Find best pose in crop
                    best_pose = None
                    best_confidence = 0
                    
                    for pose_data in crop_poses:
                        if len(pose_data) == 0:
                            continue
                        
                        keypoints = pose_data[:, :2].copy()
                        confidence = pose_data[:, 2].copy()
                        
                        valid_confidences = confidence[confidence > 0]
                        if len(valid_confidences) == 0:
                            continue
                            
                        avg_confidence = np.mean(valid_confidences)
                        if avg_confidence >= self.pose_confidence_threshold and avg_confidence > best_confidence:
                            # Adjust keypoints back to full frame
                            keypoints[:, 0] += x1_pad
                            keypoints[:, 1] += y1_pad
                            
                            best_pose = {
                                'keypoints': keypoints,
                                'confidence': confidence,
                                'from_box': True
                            }
                            best_confidence = avg_confidence
                    
                    if best_pose:
                        poses[i] = best_pose
                        box_count += 1
                        if verbose:
                            print(f"🔍 Stage 2: Found pose for detection {i} (conf: {best_confidence:.2f})")
            
            except Exception as e:
                if verbose:
                    print(f"❌ Stage 2 error for detection {i}: {e}")
                continue
        
        if verbose:
            print(f"🔍 Stage 2: {box_count} additional poses from individual boxes")
        
        # Stage 3: Dummy poses for remaining
        dummy_count = 0
        if verbose:
            print(f"🔍 Stage 3: Generating dummy poses...")
        
        for i, detection_box in enumerate(detections.xyxy):
            if poses[i] is None:
                poses[i] = self._generate_dummy_pose(detection_box)
                dummy_count += 1
        
        if verbose:
            print(f"🔍 Final: {full_frame_count} full-frame + {box_count} box + {dummy_count} dummy = {len([p for p in poses if p is not None])}/{len(detections)}")
        
        return poses
    
    def smooth_poses(self, poses, detections):
        """
        Smooth poses over time using tracking IDs to reduce flickering.
        Uses temporal averaging of keypoint positions.
        """
        if not self.enable_pose_smoothing or not hasattr(detections, 'tracker_id') or detections.tracker_id is None:
            return poses
        
        verbose = self.config.get('debug', {}).get('verbose_pose_estimation', False)
        smoothed_poses = []
        
        for i, pose in enumerate(poses):
            if pose is None or i >= len(detections.tracker_id):
                smoothed_poses.append(pose)
                continue
            
            tracker_id = int(detections.tracker_id[i])
            
            # Initialize history for new tracker
            if tracker_id not in self.pose_history:
                self.pose_history[tracker_id] = []
            
            # Add current pose to history
            self.pose_history[tracker_id].append({
                'keypoints': pose['keypoints'].copy(),
                'confidence': pose['confidence'].copy(),
                'timestamp': len(self.pose_history[tracker_id])
            })
            
            # Keep only recent history
            if len(self.pose_history[tracker_id]) > self.max_history_length:
                self.pose_history[tracker_id].pop(0)
            
            # Apply smoothing if we have enough history
            if len(self.pose_history[tracker_id]) >= 2:
                smoothed_pose = self._apply_temporal_smoothing(tracker_id, pose)
                if verbose and len(self.pose_history[tracker_id]) >= 3:
                    if i == 0:  # Only log for first detection to avoid spam
                        print(f"🔍 Smoothing: Applied to {len([p for p in poses if p is not None])} poses using {len(self.pose_history)} tracked players")
                smoothed_poses.append(smoothed_pose)
            else:
                smoothed_poses.append(pose)
        
        # Clean up old tracker histories (remove trackers not seen recently)
        self._cleanup_pose_history(detections)
        
        return smoothed_poses
    
    def _apply_temporal_smoothing(self, tracker_id, current_pose):
        """Apply temporal smoothing to a pose using its history."""
        history = self.pose_history[tracker_id]
        
        if len(history) < 2:
            return current_pose
        
        # Weights for temporal smoothing (more recent = higher weight)
        weights = np.linspace(0.3, 1.0, len(history))
        weights = weights / np.sum(weights)
        
        # Get keypoints from history
        keypoints_history = np.array([h['keypoints'] for h in history])
        confidence_history = np.array([h['confidence'] for h in history])
        
        # Smooth keypoints using weighted average
        smoothed_keypoints = np.zeros_like(current_pose['keypoints'])
        smoothed_confidence = np.zeros_like(current_pose['confidence'])
        
        for kp_idx in range(len(current_pose['keypoints'])):
            # Only smooth keypoints with sufficient confidence
            valid_mask = confidence_history[:, kp_idx] > 0.3
            
            if np.sum(valid_mask) >= 2:  # Need at least 2 valid points
                valid_keypoints = keypoints_history[valid_mask, kp_idx]
                valid_weights = weights[valid_mask]
                valid_weights = valid_weights / np.sum(valid_weights)
                
                # Weighted average of positions
                smoothed_keypoints[kp_idx] = np.average(valid_keypoints, weights=valid_weights, axis=0)
                
                # Use confidence from most recent frame
                smoothed_confidence[kp_idx] = confidence_history[-1, kp_idx]
            else:
                # Not enough valid points, use current
                smoothed_keypoints[kp_idx] = current_pose['keypoints'][kp_idx]
                smoothed_confidence[kp_idx] = current_pose['confidence'][kp_idx]
        
        # Create smoothed pose
        smoothed_pose = current_pose.copy()
        smoothed_pose['keypoints'] = smoothed_keypoints
        smoothed_pose['confidence'] = smoothed_confidence
        smoothed_pose['smoothed'] = True
        
        return smoothed_pose
    
    def _cleanup_pose_history(self, detections):
        """Remove pose history for trackers not seen recently."""
        if not hasattr(detections, 'tracker_id') or detections.tracker_id is None:
            return
        
        # Get current active tracker IDs
        current_trackers = set(int(tid) for tid in detections.tracker_id)
        
        # Remove trackers not seen in current frame (they'll be re-added if they reappear)
        trackers_to_remove = []
        for tracker_id in self.pose_history.keys():
            if tracker_id not in current_trackers:
                trackers_to_remove.append(tracker_id)
        
        for tracker_id in trackers_to_remove:
            del self.pose_history[tracker_id]
    
    def segment_players(self, frame, detections):
        """
        Enhanced player segmentation using SAM with adaptive strategies:
        - Point prompts (positive at center, negative at corners)
        - Size-based approach selection (box-based for larger players, region-based for smaller)
        """
        if self.sam_model is None or len(detections) == 0:
            return []
        
        segments = []
        original_shape = frame.shape
        frame_area = original_shape[0] * original_shape[1]
        
        # Get segmentation resolution
        seg_res = self.config.get('resolution', {}).get('segmentation', [1024, 1024])
        
        # Preprocess frame at segmentation resolution if needed
        if original_shape[0] > seg_res[1] or original_shape[1] > seg_res[0]:
            seg_frame = cv2.resize(frame, (seg_res[0], seg_res[1]))
            scale_x = original_shape[1] / seg_res[0]
            scale_y = original_shape[0] / seg_res[1]
        else:
            seg_frame = frame
            scale_x, scale_y = 1.0, 1.0
        
        # Group players by size for batch processing
        tiny_players = []
        small_players = []
        normal_players = []
        
        # Size thresholds (relative to frame area)
        tiny_threshold = 0.001  # 0.1% of frame area
        small_threshold = 0.005  # 0.5% of frame area
        
        # Process each detection
        for i, box in enumerate(detections.xyxy):
            try:
                # Extract box coordinates
                x1, y1, x2, y2 = box
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                box_ratio = box_area / frame_area
                
                # Skip very small boxes
                if box_width < 10 or box_height < 10:
                    segments.append(None)
                    continue
                    
                # Calculate center point (positive prompt)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Scale coordinates for segmentation frame
                if scale_x != 1.0 or scale_y != 1.0:
                    x1 /= scale_x
                    y1 /= scale_y
                    x2 /= scale_x
                    y2 /= scale_y
                    center_x /= scale_x
                    center_y /= scale_y
                
                # Group by size
                if box_ratio < tiny_threshold:
                    tiny_players.append((i, [x1, y1, x2, y2], [center_x, center_y]))
                elif box_ratio < small_threshold:
                    small_players.append((i, [x1, y1, x2, y2], [center_x, center_y]))
                else:
                    normal_players.append((i, [x1, y1, x2, y2], [center_x, center_y]))
                    
            except Exception as e:
                print(f"Error processing detection {i}: {e}")
                segments.append(None)
        
        # Initialize results array
        all_masks = [None] * len(detections)
        
        # Process normal-sized players individually with box+point prompts
        for i, box, center in normal_players:
            try:
                # Calculate adaptive padding
                orig_width = (box[2] - box[0]) * scale_x
                orig_height = (box[3] - box[1]) * scale_y
                pad_x, pad_y = self.calculate_adaptive_padding(orig_width, orig_height, mode='segmentation')
                pad_x /= scale_x
                pad_y /= scale_y
                
                # Create padded box
                x1_pad = max(0, box[0] - pad_x)
                y1_pad = max(0, box[1] - pad_y)
                x2_pad = min(seg_frame.shape[1], box[2] + pad_x)
                y2_pad = min(seg_frame.shape[0], box[3] + pad_y)
                
                # Create point prompts
                # Center point (positive)
                points = [center]
                # Corner points (negative)
                corner_points = [
                    [x1_pad, y1_pad],  # Top-left
                    [x2_pad, y1_pad],  # Top-right
                    [x1_pad, y2_pad],  # Bottom-left
                    [x2_pad, y2_pad]   # Bottom-right
                ]
                
                # Labels: 1 for positive (foreground), 0 for negative (background)
                labels = [1, 0, 0, 0, 0]
                
                # Run SAM with both box and point prompts
                results = self.sam_model(
                    seg_frame,
                    bboxes=[[x1_pad, y1_pad, x2_pad, y2_pad]],
                    points=[points + corner_points],
                    labels=[labels],
                    verbose=False
                )
                
                # Process results
                if (results and len(results) > 0 and
                    hasattr(results[0], 'masks') and
                    results[0].masks is not None and
                    len(results[0].masks.data) > 0):
                    
                    mask = results[0].masks.data[0].cpu().numpy()
                    
                    # Scale mask back to original resolution if needed
                    if scale_x != 1.0 or scale_y != 1.0:
                        mask = cv2.resize(
                            mask.astype(np.uint8), 
                            (original_shape[1], original_shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    
                    # Crop mask to ROI (with small margin)
                    if self.config['detection'].get('adaptive_padding', {}).get('crop_masks', True):
                        margin = 5  # pixels
                        roi_mask = np.zeros_like(mask)
                        roi_y1 = max(0, int(box[1] * scale_y) - margin)
                        roi_y2 = min(mask.shape[0], int(box[3] * scale_y) + margin)
                        roi_x1 = max(0, int(box[0] * scale_x) - margin)
                        roi_x2 = min(mask.shape[1], int(box[2] * scale_x) + margin)
                        roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = 1
                        mask = mask * roi_mask
                    
                    all_masks[i] = mask
                
            except Exception as e:
                print(f"Error in normal player segmentation for box {i}: {e}")
                all_masks[i] = None
        
        # Process small players with region-based approach (process in batches)
        if small_players:
            try:
                # Group small players into regions for batch processing
                regions = self._group_players_into_regions(small_players, seg_frame.shape)
                
                for region_players, region_box in regions:
                    if not region_players:
                        continue
                        
                    # Extract region coordinates
                    rx1, ry1, rx2, ry2 = region_box
                    
                    # Create point prompts for all players in this region
                    points = []
                    labels = []
                    
                    for _, player_box, center in region_players:
                        # Center point (positive)
                        points.append(center)
                        labels.append(1)
                    
                    # Add corner points as negative prompts
                    corner_points = [
                        [rx1, ry1],  # Top-left
                        [rx2, ry1],  # Top-right
                        [rx1, ry2],  # Bottom-left
                        [rx2, ry2]   # Bottom-right
                    ]
                    points.extend(corner_points)
                    labels.extend([0, 0, 0, 0])
                    
                    # Run SAM on the region
                    results = self.sam_model(
                        seg_frame,
                        bboxes=[region_box],
                        points=[points],
                        labels=[labels],
                        verbose=False
                    )
                    
                    # Process results - we'll get multiple masks
                    if (results and len(results) > 0 and
                        hasattr(results[0], 'masks') and
                        results[0].masks is not None and
                        len(results[0].masks.data) > 0):
                        
                        # We expect one mask per positive point (player)
                        masks = results[0].masks.data.cpu().numpy()
                        
                        # Try to match masks to players
                        for idx, (player_idx, player_box, center) in enumerate(region_players):
                            if idx < len(masks):
                                mask = masks[idx]
                                
                                # Scale mask back to original resolution if needed
                                if scale_x != 1.0 or scale_y != 1.0:
                                    mask = cv2.resize(
                                        mask.astype(np.uint8), 
                                        (original_shape[1], original_shape[0]),
                                        interpolation=cv2.INTER_NEAREST
                                    ).astype(bool)
                                
                                # Crop mask to player box (with margin)
                                if self.config['detection'].get('adaptive_padding', {}).get('crop_masks', True):
                                    margin = 5  # pixels
                                    roi_mask = np.zeros_like(mask)
                                    box_scaled = [
                                        max(0, int(player_box[0] * scale_x) - margin),
                                        max(0, int(player_box[1] * scale_y) - margin),
                                        min(mask.shape[1], int(player_box[2] * scale_x) + margin),
                                        min(mask.shape[0], int(player_box[3] * scale_y) + margin)
                                    ]
                                    roi_mask[box_scaled[1]:box_scaled[3], box_scaled[0]:box_scaled[2]] = 1
                                    mask = mask * roi_mask
                                
                                all_masks[player_idx] = mask
            except Exception as e:
                print(f"Error in small player batch segmentation: {e}")
        
        # Process tiny players with full-image SAM (if any)
        if tiny_players and len(tiny_players) > 0:
            try:
                # Create point prompts for all tiny players
                points = []
                labels = []
                
                for _, _, center in tiny_players:
                    # Center point (positive)
                    points.append(center)
                    labels.append(1)
                
                # Add negative points at edges
                edge_points = [
                    [10, 10],  # Top-left
                    [seg_frame.shape[1]-10, 10],  # Top-right
                    [10, seg_frame.shape[0]-10],  # Bottom-left
                    [seg_frame.shape[1]-10, seg_frame.shape[0]-10]  # Bottom-right
                ]
                points.extend(edge_points)
                labels.extend([0, 0, 0, 0])
                
                # Run SAM on the full image
                results = self.sam_model(
                    seg_frame,
                    points=[points],
                    labels=[labels],
                    verbose=False
                )
                
                # Process results - we'll get multiple masks
                if (results and len(results) > 0 and
                    hasattr(results[0], 'masks') and
                    results[0].masks is not None and
                    len(results[0].masks.data) > 0):
                    
                    # We expect one mask per positive point (tiny player)
                    masks = results[0].masks.data.cpu().numpy()
                    
                    # Match masks to players
                    for idx, (player_idx, player_box, _) in enumerate(tiny_players):
                        if idx < len(masks):
                            mask = masks[idx]
                            
                            # Scale mask back to original resolution if needed
                            if scale_x != 1.0 or scale_y != 1.0:
                                mask = cv2.resize(
                                    mask.astype(np.uint8), 
                                    (original_shape[1], original_shape[0]),
                                    interpolation=cv2.INTER_NEAREST
                                ).astype(bool)
                            
                            # Crop mask to player box
                            box_scaled = [
                                max(0, int(player_box[0] * scale_x)),
                                max(0, int(player_box[1] * scale_y)),
                                min(mask.shape[1], int(player_box[2] * scale_x)),
                                min(mask.shape[0], int(player_box[3] * scale_y))
                            ]
                            
                            # Create ROI mask
                            roi_mask = np.zeros_like(mask)
                            margin = 10  # pixels
                            roi_x1 = max(0, box_scaled[0] - margin)
                            roi_y1 = max(0, box_scaled[1] - margin)
                            roi_x2 = min(mask.shape[1], box_scaled[2] + margin)
                            roi_y2 = min(mask.shape[0], box_scaled[3] + margin)
                            roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = 1
                            
                            # Apply ROI
                            mask = mask * roi_mask
                            all_masks[player_idx] = mask
                            
            except Exception as e:
                print(f"Error in tiny player segmentation: {e}")
        
        # Return all masks (in original detection order)
        return all_masks

    def _group_players_into_regions(self, players, frame_shape):
        """
        Group nearby small players into regions for batch processing.
        
        Args:
            players: List of (index, box, center) tuples for small players
            frame_shape: Shape of the frame
            
        Returns:
            List of (players_in_region, region_box) tuples
        """
        if not players:
            return []
        
        # Simple algorithm: use grid-based clustering
        grid_size = 320  # pixels
        
        # Create grid cells
        grid_rows = (frame_shape[0] + grid_size - 1) // grid_size
        grid_cols = (frame_shape[1] + grid_size - 1) // grid_size
        grid = [[[] for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        # Assign players to grid cells
        for player in players:
            idx, box, center = player
            cell_row = int(center[1]) // grid_size
            cell_col = int(center[0]) // grid_size
            
            if 0 <= cell_row < grid_rows and 0 <= cell_col < grid_cols:
                grid[cell_row][cell_col].append(player)
        
        # Create regions from non-empty grid cells
        regions = []
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                cell_players = grid[row][col]
                if not cell_players:
                    continue
                
                # Calculate region bounds with padding
                padding = 20  # pixels
                min_x = max(0, col * grid_size - padding)
                min_y = max(0, row * grid_size - padding)
                max_x = min(frame_shape[1], (col + 1) * grid_size + padding)
                max_y = min(frame_shape[0], (row + 1) * grid_size + padding)
                
                region_box = [min_x, min_y, max_x, max_y]
                regions.append((cell_players, region_box))
        
        return regions
    
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