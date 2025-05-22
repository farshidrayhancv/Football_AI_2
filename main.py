#!/usr/bin/env python3
"""
Football AI - Main entry point
Refactored to use Supervision framework
"""

import argparse
import yaml
import cv2
import os
import torch
from tqdm import tqdm
from processor import FootballProcessor
from utils import get_video_info, create_video_writer


class FootballAI:
    """Main application class for Football AI."""
    
    def __init__(self, config_path):
        """Initialize with configuration."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup environment variables
        os.environ["ROBOFLOW_API_KEY"] = self.config['api_keys']['roboflow_api_key']
        
        # Setup device
        device = self.config['performance']['device']
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU instead")
            device = 'cpu'
        
        # Initialize processor
        self.processor = FootballProcessor(self.config, device)
        
    def process_video(self, input_path=None, output_path=None):
        """Process video with Football AI."""
        # Use paths from config if not specified
        if input_path is None:
            input_path = self.config['video']['input_path']
        if output_path is None:
            output_path = self.config['video']['output_path']
            
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get video properties
        cap, width, height, fps, total_frames = get_video_info(input_path)
        if cap is None:
            print(f"Error: Could not open video {input_path}")
            return
        
        # Process resolution
        proc_width, proc_height = self.config['processing']['resolution']
        out_width, out_height = self.config['processing']['output_resolution']
        
        # Create video writer
        writer = create_video_writer(output_path, out_width, out_height, fps)
        
        # Process frames
        stride = self.config['video']['stride']
        frame_idx = 0
        
        with tqdm(total=total_frames//stride, desc="Processing") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every nth frame based on stride
                if frame_idx % stride != 0:
                    frame_idx += 1
                    continue
                
                # Resize frame for processing if needed
                if (width, height) != (proc_width, proc_height):
                    proc_frame = cv2.resize(frame, (proc_width, proc_height))
                else:
                    proc_frame = frame
                
                # Process frame
                result_frame = self.processor.process_frame(proc_frame)
                
                # Resize to output resolution if needed
                if result_frame.shape[:2] != (out_height, out_width):
                    result_frame = cv2.resize(result_frame, (out_width, out_height))
                
                # Write frame
                writer.write(result_frame)
                
                # Update progress
                frame_idx += 1
                pbar.update(1)
                
                # Save preview image occasionally
                if frame_idx % 100 == 0:
                    preview_path = output_path.replace('.mp4', f'_preview_{frame_idx}.jpg')
                    cv2.imwrite(preview_path, result_frame)
        
        # Release resources
        cap.release()
        writer.release()
        
        print(f"Processing complete! Output saved to: {output_path}")
        

def main():
    """Parse arguments and run application."""
    parser = argparse.ArgumentParser(description='Football AI Application')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input video (overrides config)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output video (overrides config)')
    args = parser.parse_args()
    
    # Initialize and run application
    app = FootballAI(args.config)
    app.process_video(args.input, args.output)


if __name__ == "__main__":
    main()
