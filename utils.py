"""
Utility functions for Football AI
"""

import cv2
import os
import yaml
import numpy as np


def get_video_info(video_path):
    """
    Get video properties.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Tuple of (cap, width, height, fps, total_frames)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None, 0, 0, 0, 0
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    return cap, width, height, fps, total_frames


def create_video_writer(output_path, width, height, fps):
    """
    Create video writer.
    
    Args:
        output_path: Path to output video file
        width: Width of output video
        height: Height of output video
        fps: Frames per second
    
    Returns:
        OpenCV VideoWriter object
    """
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def combine_images(images, layout=(1, 2)):
    """
    Combine multiple images into a grid.
    
    Args:
        images: List of images to combine
        layout: Tuple of (rows, cols)
    
    Returns:
        Combined image
    """
    if not images:
        return None
    
    rows, cols = layout
    
    # Find max dimensions
    max_height = max(img.shape[0] for img in images if img is not None)
    max_width = max(img.shape[1] for img in images if img is not None)
    
    # Create output image
    output = np.zeros((max_height * rows, max_width * cols, 3), dtype=np.uint8)
    
    # Place images
    for i, img in enumerate(images):
        if img is None:
            continue
            
        r = i // cols
        c = i % cols
        
        y1 = r * max_height
        y2 = y1 + img.shape[0]
        x1 = c * max_width
        x2 = x1 + img.shape[1]
        
        output[y1:y2, x1:x2] = img
    
    return output


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Config dictionary
    """
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}
