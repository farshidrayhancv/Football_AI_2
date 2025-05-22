#!/usr/bin/env python3
"""
Football AI V2 - Simplified Version
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
    parser = argparse.ArgumentParser(description='Football AI V2')
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
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize processor
    processor = FootballProcessor(config, device)
    
    # Process video
    processor.process_video()


if __name__ == "__main__":
    main()
