"""
Football AI V2 - Model Management
Handles Roboflow, YOLO, and SAM models
Now with support for custom YOLO field detection and adaptive padding
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
    
    def estimate_poses(self, frame, detections):
        """Estimate poses for detected humans with adaptive padding."""
        if self.pose_model is None or len(detections) == 0:
            return []
        
        poses = []
        
        for i, box in enumerate(detections.xyxy):
            try:
                # Extract box coordinates
                x1, y1, x2, y2 = map(int, box)
                box_width = x2 - x1
                box_height = y2 - y1
                
                # Skip very small boxes that might be false detections
                if box_width < 10 or box_height < 10:
                    poses.append(None)
                    continue
                
                # Calculate adaptive padding based on box size
                pad_x, pad_y = self.calculate_adaptive_padding(box_width, box_height, mode='pose')
                
                # Apply padding with frame boundaries check
                x1_pad = max(0, x1 - pad_x)
                y1_pad = max(0, y1 - pad_y)
                x2_pad = min(frame.shape[1], x2 + pad_x)
                y2_pad = min(frame.shape[0], y2 + pad_y)
                
                # Extract crop
                crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
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
                        'confidence': kpts[:, 2].copy(),
                        'box_padding': (pad_x, pad_y),  # Store padding info for debugging
                        'box_size': (box_width, box_height)
                    }
                    
                    pose_data['keypoints'][:, 0] += x1_pad
                    pose_data['keypoints'][:, 1] += y1_pad
                    
                    # Validate keypoints are within frame bounds
                    pose_data['keypoints'][:, 0] = np.clip(pose_data['keypoints'][:, 0], 0, frame.shape[1])
                    pose_data['keypoints'][:, 1] = np.clip(pose_data['keypoints'][:, 1], 0, frame.shape[0])
                    
                    poses.append(pose_data)
                else:
                    poses.append(None)
                    
            except Exception as e:
                print(f"Error in pose estimation for box {i}: {e}")
                poses.append(None)
        
        return poses
    
    def segment_players(self, frame, detections):
        """Segment players using SAM with adaptive padding."""
        if self.sam_model is None or len(detections) == 0:
            return []
        
        segments = []
        
        for i, box in enumerate(detections.xyxy):
            try:
                # Extract box coordinates
                x1, y1, x2, y2 = box
                box_width = x2 - x1
                box_height = y2 - y1
                
                # Skip very small boxes
                if box_width < 10 or box_height < 10:
                    segments.append(None)
                    continue
                
                # Calculate adaptive padding for segmentation
                pad_x, pad_y = self.calculate_adaptive_padding(box_width, box_height, mode='segmentation')
                
                # Create padded box for SAM
                x1_pad = max(0, x1 - pad_x)
                y1_pad = max(0, y1 - pad_y)
                x2_pad = min(frame.shape[1], x2 + pad_x)
                y2_pad = min(frame.shape[0], y2 + pad_y)
                
                padded_box = [x1_pad, y1_pad, x2_pad, y2_pad]
                
                # Run SAM with padded box prompt
                # Note: Some SAM implementations work better with slightly tighter boxes
                # You can adjust the padding specifically for SAM if needed
                sam_padding_factor = self.config['detection'].get('adaptive_padding', {}).get('sam_padding_factor', 0.8)
                sam_pad_x = int(pad_x * sam_padding_factor)
                sam_pad_y = int(pad_y * sam_padding_factor)
                
                sam_box = [
                    max(0, x1 - sam_pad_x),
                    max(0, y1 - sam_pad_y),
                    min(frame.shape[1], x2 + sam_pad_x),
                    min(frame.shape[0], y2 + sam_pad_y)
                ]
                
                results = self.sam_model(
                    frame,
                    bboxes=[sam_box],
                    verbose=False
                )
                
                if (results and len(results) > 0 and
                    hasattr(results[0], 'masks') and
                    results[0].masks is not None and
                    len(results[0].masks.data) > 0):
                    
                    mask = results[0].masks.data[0].cpu().numpy()
                    
                    # Optionally crop the mask to remove padding artifacts
                    # This ensures the mask focuses on the player, not the padding area
                    if self.config['detection'].get('adaptive_padding', {}).get('crop_masks', True):
                        # Create a region of interest based on original box + small margin
                        margin = 5  # pixels
                        roi_mask = np.zeros_like(mask)
                        roi_y1 = max(0, int(y1) - margin)
                        roi_y2 = min(mask.shape[0], int(y2) + margin)
                        roi_x1 = max(0, int(x1) - margin)
                        roi_x2 = min(mask.shape[1], int(x2) + margin)
                        roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = 1
                        
                        # Apply ROI to mask
                        mask = mask * roi_mask
                    
                    segments.append(mask)
                else:
                    segments.append(None)
                    
            except Exception as e:
                print(f"Error in segmentation for box {i}: {e}")
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