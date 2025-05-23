#!/usr/bin/env python3
"""
Football AI V2 - Simple Version with Per-Model Resolution Control
Main entry point
"""

import argparse
import yaml
import cv2
import os
import torch
from processor import FootballProcessor


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set environment variables
    os.environ["ROBOFLOW_API_KEY"] = config['api_keys']['roboflow_api_key']
    if config['api_keys'].get('huggingface_token'):
        os.environ["HF_TOKEN"] = config['api_keys']['huggingface_token']
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Football AI V2 - Simple Resolution Control')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--input', type=str, default=None,
                        help='Input video path (overrides config)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output video path (overrides config)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override paths if provided
    if args.input:
        config['video']['input_path'] = args.input
    if args.output:
        config['video']['output_path'] = args.output
    
    # Display configuration
    print("Football AI V2 - Multi-Resolution Processing")
    print("=" * 50)
    
    resolutions = config.get('resolution', {})
    print("Resolution Configuration:")
    print(f"  Object Detection: {resolutions.get('object_detection', [640, 640])}")
    print(f"  Pose Estimation:  {resolutions.get('pose_estimation', [480, 480])}")
    print(f"  Segmentation:     {resolutions.get('segmentation', [512, 512])}")
    print(f"  Field Detection:  {resolutions.get('field_detection', [1280, 720])}")
    print(f"  Output Video:     {resolutions.get('output', [800, 600])}")
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nCompute Device: {device.upper()}")
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    
    # Check video file
    video_path = config['video']['input_path']
    if not os.path.exists(video_path):
        print(f"\n❌ Video file not found: {video_path}")
        return 1
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\n❌ Cannot open video: {video_path}")
        return 1
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"\nVideo Information:")
    print(f"  File: {os.path.basename(video_path)}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.1f} seconds")
    
    # Initialize processor
    print(f"\nInitializing Football AI V2...")
    try:
        processor = FootballProcessor(config, device)
    except Exception as e:
        print(f"❌ Failed to initialize processor: {e}")
        if "out of memory" in str(e).lower():
            print("\n💡 Suggestion: Try reducing the resolution settings in your config:")
            print("   resolution:")
            print("     object_detection: [480, 480]  # Reduce from [640, 640]")
            print("     pose_estimation: [320, 320]   # Reduce from [480, 480]")
            print("     segmentation: [384, 384]      # Reduce from [512, 512]")
        return 1
    
    # Process video
    print(f"\nStarting video processing...")
    try:
        processor.process_video()
        print(f"\n✅ Processing completed successfully!")
        
        # Display output information
        output_path = config['video']['output_path']
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"Output: {output_path}")
            print(f"Size: {file_size:.1f} MB")
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)