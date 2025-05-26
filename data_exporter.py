#!/usr/bin/env python3
"""
Football AI V2 - Data Exporter
Exports detection data with both frame and pitch coordinates for Blender visualization
"""

import json
import numpy as np
import os
from pathlib import Path
import supervision as sv


class FootballDataExporter:
    """Exports football analysis data to JSON for external visualization."""
    
    def __init__(self, config, output_path=None):
        """Initialize data exporter."""
        self.config = config
        self.export_config = config.get('export', {})
        self.enabled = self.export_config.get('enable', False)
        
        if not self.enabled:
            return
            
        # Set output path
        if output_path:
            self.output_path = output_path
        else:
            self.output_path = self.export_config.get('output_path', 'output/analysis_data.json')
        
        # Create output directory
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # Initialize data storage
        self.frame_data = []
        self.video_metadata = {}
        
        print(f"✓ Data exporter initialized - Output: {self.output_path}")
    
    def set_video_metadata(self, width, height, fps, total_frames):
        """Set video metadata."""
        if not self.enabled:
            return
            
        self.video_metadata = {
            'width': int(width),
            'height': int(height),
            'fps': float(fps),
            'total_frames': int(total_frames),
            'export_timestamp': str(np.datetime64('now'))
        }
    
    def export_frame_data(self, frame_number, detections, poses, field_keypoints, transformer, possession_info=None):
        """Export data for a single frame."""
        if not self.enabled:
            return
        
        frame_data = {
            'frame_number': int(frame_number),
            'timestamp': float(frame_number / self.video_metadata.get('fps', 25.0)),
            'field_keypoints': self._export_field_keypoints(field_keypoints, transformer),
            'detections': self._export_detections(detections, transformer),
            'poses': self._export_poses(poses, detections, transformer),
            'possession': self._export_possession(possession_info)
        }
        
        self.frame_data.append(frame_data)
    
    def _export_field_keypoints(self, field_keypoints, transformer):
        """Export field keypoints with both frame and pitch coordinates."""
        if field_keypoints is None or len(field_keypoints.xy) == 0:
            return None
        
        keypoints_data = {
            'frame_coordinates': [],
            'pitch_coordinates': [],
            'confidence': [],
            'detected_count': 0
        }
        
        # Get keypoints and confidence
        all_keypoints = field_keypoints.xy[0]
        all_confidences = field_keypoints.confidence[0]
        
        # Get confidence threshold
        confidence_threshold = self.config['detection']['keypoint_confidence_threshold']
        
        # Export all keypoints (both high and low confidence for reference)
        for i, (keypoint, confidence) in enumerate(zip(all_keypoints, all_confidences)):
            keypoints_data['frame_coordinates'].append({
                'index': i,
                'x': float(keypoint[0]),
                'y': float(keypoint[1]),
                'confidence': float(confidence),
                'high_confidence': bool(confidence > confidence_threshold)
            })
        
        # Add pitch coordinates (these are the known pitch vertices)
        if transformer is not None:
            from sports.configs.soccer import SoccerPitchConfiguration
            pitch_config = SoccerPitchConfiguration()
            pitch_vertices = np.array(pitch_config.vertices)
            
            for i, vertex in enumerate(pitch_vertices):
                keypoints_data['pitch_coordinates'].append({
                    'index': i,
                    'x': float(vertex[0]),
                    'y': float(vertex[1]),
                    'description': self._get_keypoint_description(i)
                })
        
        keypoints_data['detected_count'] = int(np.sum(all_confidences > confidence_threshold))
        
        return keypoints_data
    
    def _export_detections(self, detections, transformer):
        """Export all detections with frame and pitch coordinates."""
        detection_data = {}
        
        for category in ['players', 'goalkeepers', 'referees', 'ball']:
            category_detections = detections[category]
            category_data = []
            
            if len(category_detections) == 0:
                detection_data[category] = category_data
                continue
            
            # Get positions in frame coordinates
            if category == 'ball':
                frame_positions = category_detections.get_anchors_coordinates(sv.Position.CENTER)
            else:
                frame_positions = category_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            
            # Transform to pitch coordinates if possible
            pitch_positions = None
            if transformer is not None and len(frame_positions) > 0:
                try:
                    pitch_positions = transformer.transform_points(frame_positions)
                except:
                    pitch_positions = None
            
            # Export each detection
            for i in range(len(category_detections)):
                detection_item = {
                    'detection_id': i,
                    'confidence': float(category_detections.confidence[i]) if category_detections.confidence is not None else 1.0,
                    'bounding_box': {
                        'x1': float(category_detections.xyxy[i][0]),
                        'y1': float(category_detections.xyxy[i][1]),
                        'x2': float(category_detections.xyxy[i][2]),
                        'y2': float(category_detections.xyxy[i][3])
                    },
                    'frame_position': {
                        'x': float(frame_positions[i][0]),
                        'y': float(frame_positions[i][1])
                    },
                    'pitch_position': None,
                    'tracking_id': None,
                    'team_id': None
                }
                
                # Add pitch position if available
                if pitch_positions is not None and i < len(pitch_positions):
                    detection_item['pitch_position'] = {
                        'x': float(pitch_positions[i][0]),
                        'y': float(pitch_positions[i][1])
                    }
                
                # Add tracking ID if available
                if hasattr(category_detections, 'tracker_id') and category_detections.tracker_id is not None:
                    if i < len(category_detections.tracker_id):
                        detection_item['tracking_id'] = int(category_detections.tracker_id[i])
                
                # Add team ID for players/goalkeepers
                if category in ['players', 'goalkeepers']:
                    if hasattr(category_detections, 'class_id') and category_detections.class_id is not None:
                        if i < len(category_detections.class_id):
                            detection_item['team_id'] = int(category_detections.class_id[i])
                
                category_data.append(detection_item)
            
            detection_data[category] = category_data
        
        return detection_data
    
    def _export_poses(self, poses, detections, transformer):
        """Export pose data with frame and pitch coordinates."""
        if not poses:
            return {}
        
        pose_data = {}
        
        for category in ['players', 'goalkeepers', 'referees']:
            if category not in poses:
                pose_data[category] = []
                continue
            
            category_poses = poses[category]
            category_data = []
            
            for i, pose in enumerate(category_poses):
                if pose is None:
                    category_data.append(None)
                    continue
                
                pose_item = {
                    'detection_id': i,
                    'keypoints_frame': [],
                    'keypoints_pitch': [],
                    'tracking_id': None,
                    'team_id': None,
                    'pose_confidence': float(np.mean(pose['confidence'][pose['confidence'] > 0])) if len(pose['confidence'][pose['confidence'] > 0]) > 0 else 0.0
                }
                
                # Export frame keypoints
                for j, (keypoint, confidence) in enumerate(zip(pose['keypoints'], pose['confidence'])):
                    pose_item['keypoints_frame'].append({
                        'joint_id': j,
                        'joint_name': self._get_coco_joint_name(j),
                        'x': float(keypoint[0]),
                        'y': float(keypoint[1]),
                        'confidence': float(confidence),
                        'visible': bool(confidence > 0.5)
                    })
                
                # Transform keypoints to pitch coordinates if possible
                if transformer is not None:
                    try:
                        visible_keypoints = pose['keypoints'][pose['confidence'] > 0.5]
                        if len(visible_keypoints) > 0:
                            pitch_keypoints = transformer.transform_points(visible_keypoints)
                            
                            visible_indices = np.where(pose['confidence'] > 0.5)[0]
                            for idx, (pitch_kp, orig_idx) in enumerate(zip(pitch_keypoints, visible_indices)):
                                pose_item['keypoints_pitch'].append({
                                    'joint_id': int(orig_idx),
                                    'joint_name': self._get_coco_joint_name(orig_idx),
                                    'x': float(pitch_kp[0]),
                                    'y': float(pitch_kp[1]),
                                    'confidence': float(pose['confidence'][orig_idx])
                                })
                    except:
                        pass  # Skip transformation if it fails
                
                # Add tracking and team info
                if len(detections[category]) > i:
                    category_detections = detections[category]
                    
                    if hasattr(category_detections, 'tracker_id') and category_detections.tracker_id is not None:
                        if i < len(category_detections.tracker_id):
                            pose_item['tracking_id'] = int(category_detections.tracker_id[i])
                    
                    if category in ['players', 'goalkeepers']:
                        if hasattr(category_detections, 'class_id') and category_detections.class_id is not None:
                            if i < len(category_detections.class_id):
                                pose_item['team_id'] = int(category_detections.class_id[i])
                
                category_data.append(pose_item)
            
            pose_data[category] = category_data
        
        return pose_data
    
    def _export_possession(self, possession_info):
        """Export possession information."""
        if possession_info is None:
            return None
        
        return {
            'player_id': possession_info.get('player_id'),
            'category': possession_info.get('category'),
            'distance': float(possession_info.get('distance', 0.0))
        }
    
    def _get_keypoint_description(self, index):
        """Get description for pitch keypoint."""
        descriptions = [
            "Pitch corner (top-left)",
            "Pitch corner (top-right)", 
            "Pitch corner (bottom-right)",
            "Pitch corner (bottom-left)",
            "Penalty area (top-left)",
            "Penalty area (top-right)",
            "Penalty area (bottom-right)",
            "Penalty area (bottom-left)",
            "Goal area (top-left)",
            "Goal area (top-right)",
            "Goal area (bottom-right)",
            "Goal area (bottom-left)",
            "Center circle (top)",
            "Center circle (right)",
            "Center circle (bottom)",
            "Center circle (left)",
            "Center spot",
            "Penalty spot (left)",
            "Penalty spot (right)"
        ]
        
        if index < len(descriptions):
            return descriptions[index]
        else:
            return f"Keypoint {index}"
    
    def _get_coco_joint_name(self, index):
        """Get COCO pose joint names."""
        joint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        if index < len(joint_names):
            return joint_names[index]
        else:
            return f"joint_{index}"
    
    def save_data(self):
        """Save all collected data to JSON file."""
        if not self.enabled:
            return
        
        if not self.frame_data:
            print("Warning: No frame data to export")
            return
        
        export_data = {
            'metadata': self.video_metadata,
            'config': {
                'field_detection_model': self.config['models'].get('field_detection_model_id', 'unknown'),
                'player_detection_model': self.config['models'].get('player_detection_model_id', 'unknown'),
                'confidence_threshold': self.config['detection']['confidence_threshold'],
                'keypoint_confidence_threshold': self.config['detection']['keypoint_confidence_threshold']
            },
            'frames': self.frame_data,
            'statistics': {
                'total_frames': len(self.frame_data),
                'average_players_per_frame': self._calculate_average_detections('players'),
                'average_ball_detections_per_frame': self._calculate_average_detections('ball'),
                'frames_with_field_detection': sum(1 for frame in self.frame_data if frame['field_keypoints'] is not None)
            }
        }
        
        # Save to JSON
        with open(self.output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=self._json_serializer)
        
        print(f"✅ Analysis data exported to: {self.output_path}")
        print(f"   Exported {len(self.frame_data)} frames")
        print(f"   File size: {os.path.getsize(self.output_path) / (1024*1024):.1f} MB")
    
    def _calculate_average_detections(self, category):
        """Calculate average number of detections per frame for a category."""
        if not self.frame_data:
            return 0.0
        
        total_detections = sum(len(frame['detections'][category]) for frame in self.frame_data)
        return total_detections / len(self.frame_data)
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")